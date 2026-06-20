import torch
import pandas as pd
import numpy as np
from pathlib import Path

from config import Config
from src.data.dataset import load_sc_waveform, extract_chunk


SR = 32000


def load_val_waveforms(cfg: Config, val_sc_df: pd.DataFrame):
    sc_file_meta = pd.read_csv(
        cfg.waveform_cache_path / "soundscape_file_meta.csv"
    )
    sc_file_dict = dict(zip(sc_file_meta["filename"], sc_file_meta["cache_file"]))
    wavs = []
    for _, row in val_sc_df.iterrows():
        cf = sc_file_dict.get(row["filename"])
        if cf is not None:
            w = load_sc_waveform(cfg.waveform_cache_path, cf)
            if w is not None:
                chunk = extract_chunk(
                    w, int(row["start_sec"]) * SR, cfg.val_samples
                )
                wavs.append(chunk.float().unsqueeze(0))
            else:
                wavs.append(torch.zeros(1, cfg.val_samples, dtype=torch.float32))
        else:
            wavs.append(torch.zeros(1, cfg.val_samples, dtype=torch.float32))
    return wavs


def predict_from_waveforms(
    model: torch.nn.Module,
    mel_transform: torch.nn.Module,
    wav_list: list,
    device: torch.device,
    batch_size: int = 64,
):
    model.eval()
    preds_clip, preds_fmax, preds_blend = [], [], []
    with torch.no_grad():
        for s in range(0, len(wav_list), batch_size):
            batch = torch.stack(wav_list[s : s + batch_size]).to(device, non_blocking=True)
            mel = mel_transform(batch)
            mean = mel.mean(dim=(2, 3), keepdim=True)
            std = mel.std(dim=(2, 3), keepdim=True) + 1e-6
            mel = (mel - mean) / std
            with torch.amp.autocast(device_type="cuda"):
                clip_logits, framewise = model(mel, return_framewise=True)
                frame_max = framewise.max(dim=1).values
                p_clip = torch.sigmoid(clip_logits)
                p_fmax = torch.sigmoid(frame_max)
                p_blend = 0.5 * p_clip + 0.5 * p_fmax
            preds_clip.append(p_clip.cpu())
            preds_fmax.append(p_fmax.cpu())
            preds_blend.append(p_blend.cpu())
    return {
        "clip": torch.cat(preds_clip).numpy(),
        "fmax": torch.cat(preds_fmax).numpy(),
        "blend": torch.cat(preds_blend).numpy(),
    }
