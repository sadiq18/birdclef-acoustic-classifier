import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import timm
from pathlib import Path

from config import Config
from .sed_model import GeMFreqPool


class SEDExportWrapper(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=False,
            in_chans=1,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.1,
        )
        with torch.no_grad():
            n_tf = cfg.val_samples // cfg.hop_length + 1
            dummy = torch.randn(1, 1, cfg.n_mels, n_tf)
            feat = self.backbone(dummy)
            backbone_dim = feat.shape[1]

        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense_drop1 = nn.Dropout(0.25)
        self.dense_conv = nn.Conv1d(backbone_dim, cfg.hidden_dim, 1)
        self.dense_relu = nn.ReLU(inplace=True)
        self.dense_drop2 = nn.Dropout(0.5)
        self.att = nn.Conv1d(cfg.hidden_dim, cfg.num_classes, 1)
        self.cla = nn.Conv1d(cfg.hidden_dim, cfg.num_classes, 1)

    def forward(self, mel):
        h = self.backbone(mel)
        h = self.gem_freq(h)
        h = self.dense_drop1(h)
        h = self.dense_conv(h)
        h = self.dense_relu(h)
        h = self.dense_drop2(h)
        att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
        framewise = self.cla(h)
        clip = torch.sum(att * framewise, dim=2)
        return clip, framewise.permute(0, 2, 1)


def load_and_remap_state(export_model: SEDExportWrapper, trained_state: dict):
    remap = {}
    for k, v in trained_state.items():
        if k.startswith("distill_head."):
            continue
        if k == "dense.1.weight":
            remap["dense_conv.weight"] = v.unsqueeze(-1)
        elif k == "dense.1.bias":
            remap["dense_conv.bias"] = v
        else:
            remap[k] = v
    export_model.load_state_dict(remap, strict=False)


def export_to_onnx(
    model: nn.Module,
    trained_state: dict,
    onnx_path: Path,
    cfg: Config,
):
    n_frames = cfg.val_samples // cfg.hop_length + 1
    dummy_mel = torch.randn(1, 1, cfg.n_mels, n_frames)

    export_model = SEDExportWrapper(cfg)
    load_and_remap_state(export_model, trained_state)
    export_model.eval()

    torch.onnx.export(
        export_model,
        dummy_mel,
        str(onnx_path),
        input_names=["mel"],
        output_names=["clip_logits", "framewise_logits"],
        dynamic_axes={
            "mel": {0: "batch"},
            "clip_logits": {0: "batch"},
            "framewise_logits": {0: "batch"},
        },
        opset_version=18,
    )

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_out = sess.run(None, {"mel": dummy_mel.numpy()})
    with torch.no_grad():
        ref_clip, _ = export_model(dummy_mel)
    diff = np.abs(ref_clip.numpy() - onnx_out[0]).max()
    return diff
