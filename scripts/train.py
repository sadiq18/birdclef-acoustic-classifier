#!/usr/bin/env python3
"""
BirdCLEF 2026 — Full Training Pipeline

Usage:
    python scripts/train.py [--debug] [--folds 0 1 2 3 4] [--epochs 10]

Runs k-fold CV training with:
    - Focal recordings + soundscape data
    - EfficientNet backbone with GeMFreq pooling + attention
    - Optional Perch distillation (MSE teacher)
    - ONNX export & verification per fold
    - OOF evaluation across all folds
"""

import argparse
import json
import pickle
import time
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.data.loader import load_data, get_fold_data
from src.training.train import train_fold
from src.training.metrics import compute_macro_auc, full_eval
from src.models.mel import MelSpecTransform
from src.models.sed_model import make_model
from src.models.export import export_to_onnx
from src.training.val import load_val_waveforms, predict_from_waveforms


def parse_args():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Training Pipeline")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 1 epoch, 1 fold, tiny data")
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Folds to train (e.g. 0 1 2 3 4)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    if args.debug:
        cfg.debug = True
        cfg.epochs = 1
        cfg.folds = [0]
    if args.folds is not None:
        cfg.folds = args.folds
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Backbone: {cfg.backbone_name}")
    print(f"Epochs: {cfg.epochs} | Batch: {cfg.batch_size} | Folds: {cfg.folds}")
    print(f"Debug: {cfg.debug}")
    print("=" * 60)

    run_dir = cfg.out_path / time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_link = cfg.out_path / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir, target_is_directory=True)

    data = load_data(cfg)
    print(f"\nFocal samples: {len(data['audio_cache_meta'])}")
    print(f"Soundscape windows: {len(data['sc_cache_meta'])}")

    if cfg.debug:
        data["audio_cache_meta"] = (
            data["audio_cache_meta"].groupby("primary_label").head(3).reset_index(drop=True)
        )
        data["sc_cache_meta"] = data["sc_cache_meta"].head(50)
        data["Y_SC"] = data["Y_SC"][:50]
        data["non_s22_mask_sc"] = data["non_s22_mask_sc"][:50]

    oof_ns22 = np.full(
        (len(data["sc_cache_meta"]), cfg.num_classes), np.nan, dtype=np.float32
    )
    all_histories = {}

    for fold_k in cfg.folds:
        print("\n" + "=" * 60)
        print(f"FOLD {fold_k}")
        print("=" * 60)

        best_ns22_state, best_macro_state, history = train_fold(
            cfg, data, fold_k, device
        )
        all_histories[fold_k] = history

        fold_data = get_fold_data(data, fold_k)

        if best_macro_state is None:
            continue

        ckpt_ns22 = run_dir / f"fold{fold_k}_best_ns22.pt"
        ckpt_macro = run_dir / f"fold{fold_k}_best_macro.pt"
        torch.save(best_ns22_state, ckpt_ns22)
        torch.save(best_macro_state, ckpt_macro)

        mel_tf = MelSpecTransform(cfg).to(device)
        val_wavs_k = load_val_waveforms(cfg, fold_data["val_sc_df"])

        m = make_model(cfg, device)
        m.load_state_dict(best_macro_state, strict=False)
        m.eval()
        with torch.no_grad():
            oof_ns22[fold_data["val_sc_mask"]] = predict_from_waveforms(
                m, mel_tf, val_wavs_k, device
            )["blend"]

        onnx_path = run_dir / f"sed_distill_fold{fold_k}.onnx"
        diff = export_to_onnx(m, best_macro_state, onnx_path, cfg)
        print(f"  ONNX verify: max|diff|={diff:.3e}")
        print(f"  Exported {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

        del m, mel_tf
        torch.cuda.empty_cache()

    has_oof = ~np.isnan(oof_ns22[:, 0])
    if has_oof.sum() > 0:
        print("\n" + "=" * 60)
        print("OOF RESULTS (best-macro checkpoints)")
        print("=" * 60)
        r_all = full_eval(
            data["Y_SC"][has_oof],
            oof_ns22[has_oof],
            data["non_s22_mask_sc"][has_oof],
            data["taxon_masks"],
        )
        print(f"  macro AUC (all):        {r_all['macro_auc_all']:.4f}")
        print(f"  macro AUC (non-S22):    {r_all['non_s22_macro']:.4f}")
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            print(f"    {t:<12}: {r_all.get(f'non_s22_{t}', float('nan')):.4f}")

        print("\nPer-epoch pooled non-S22 AUC:")
        fold_true, fold_ns22_m = {}, {}
        for fk in cfg.folds:
            vm = data["sc_cache_meta"]["fold"].values == fk
            fold_true[fk] = data["Y_SC"][vm]
            fold_ns22_m[fk] = data["non_s22_mask_sc"][vm]
        n_eps = [
            len(all_histories[k]["val_preds"])
            for k in cfg.folds if k in all_histories
        ]
        max_ep = min(n_eps) if n_eps else 0
        for ep in range(max_ep):
            pp = np.concatenate([
                all_histories[k]["val_preds"][ep]
                for k in cfg.folds if k in all_histories
            ])
            pt = np.concatenate([
                fold_true[k] for k in cfg.folds if k in all_histories
            ])
            pm = np.concatenate([
                fold_ns22_m[k] for k in cfg.folds if k in all_histories
            ])
            ns, _ = compute_macro_auc(pt, pp, mask=pm)
            print(f"  Ep{ep:02d}: {ns:.4f}")

        oof_path = run_dir / "oof_predictions.npy"
        np.save(oof_path, oof_ns22)

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

    hist_path = run_dir / "histories.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(all_histories, f)

    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
