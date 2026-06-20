import torch
import torch.nn as nn
import timm

from config import Config


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)

    def forward(self, feature_map):
        gap = feature_map.mean(dim=[2, 3])
        return self.proj(gap)


class BirdSEDModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool="",
            drop_path_rate=cfg.drop_path_rate,
        )
        with torch.no_grad():
            n_tf = cfg.train_samples // cfg.hop_length + 1
            dummy = torch.randn(1, 1, cfg.n_mels, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]

        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(cfg.hidden_dim, cfg.num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(cfg.hidden_dim, cfg.num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.xavier_uniform_(self.cla.weight)
        self.att.bias.data.fill_(0.0)
        self.cla.bias.data.fill_(0.0)

        if cfg.use_perch_distill:
            self.distill_head = DistillHead(self.backbone_dim, cfg.perch_embed_dim)

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)
        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)

        h_cls = h.detach() if self.cfg.use_perch_distill else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)

        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)

        fw = framewise_logits.permute(0, 2, 1) if return_framewise else None
        if return_framewise and return_distill:
            return clip_logits, fw, distill_emb
        elif return_framewise:
            return clip_logits, fw
        elif return_distill:
            return clip_logits, distill_emb
        return clip_logits


def make_model(cfg: Config, device: torch.device):
    return BirdSEDModel(cfg).to(device)
