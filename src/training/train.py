import time
import gc
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler

from config import Config
from src.data.dataset import FocalDS, ScDS, MixSamp, collate_m
from src.data.loader import get_fold_data
from src.models.mel import MelSpecTransform, SpecAugment
from src.models.sed_model import make_model
from src.models.perch import PerchTeacher
from src.models.export import export_to_onnx
from src.training.val import load_val_waveforms, predict_from_waveforms
from src.training.metrics import full_eval


SOURCE_WEIGHTS = {"focal": 1.0, "focal_missing": 0.0, "sc": 1.0}


def build_active_datasets(cfg: Config, data: dict, fold_k: int):
    items = []
    if cfg.use_focal:
        focal_df = data["audio_cache_meta"][
            data["audio_cache_meta"]["fold"] != fold_k
        ]
        fds = FocalDS(
            df=focal_df,
            label2idx=data["label2idx"],
            cache_dir=cfg.waveform_cache_path,
            train_samples=cfg.train_samples,
            num_classes=cfg.num_classes,
            cfg=cfg,
            secondary_lookup=data["secondary_labels"],
            aug=True,
        )
        items.append(("focal", fds, len(fds)))
    if cfg.use_labeled_sc:
        vm = data["sc_cache_meta"]["fold"].values == fold_k
        sc_train_df = data["sc_cache_meta"].loc[~vm]
        Y_tr = data["Y_SC"][~vm]
        sds = ScDS(
            Y=Y_tr,
            sc_df=sc_train_df,
            cache_dir=cfg.waveform_cache_path,
            train_samples=cfg.train_samples,
            num_classes=cfg.num_classes,
            cfg=cfg,
            aug=True,
        )
        items.append(("sc", sds, len(sds)))
    return items


def train_fold(
    cfg: Config,
    data: dict,
    fold_k: int,
    device: torch.device,
):
    fold_data = get_fold_data(data, fold_k)

    active = build_active_datasets(cfg, data, fold_k)
    names, datasets, sizes = zip(*active)

    mds = ConcatDataset(list(datasets))
    num_steps = max(100, int(sum(sizes) / cfg.batch_size))

    model = make_model(cfg, device)
    mel_transform = MelSpecTransform(cfg).to(device)
    spec_augment = SpecAugment(cfg).to(device)

    perch_teacher = None
    if cfg.use_perch_distill:
        perch_teacher = PerchTeacher(
            cfg.perch_onnx,
            "cuda" if torch.cuda.is_available() else "cpu",
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = GradScaler()

    warmup_steps = num_steps * cfg.warmup_epochs
    total_steps = num_steps * cfg.epochs

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1 / 25, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=1e-6
    )
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    history = {
        "ep": [],
        "train_loss": [],
        "cls_loss": [],
        "dist_loss": [],
        "macro": [],
        "ns22_macro": [],
    }
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
        history[f"ns22_{t}"] = []
    history["val_preds"] = []

    best_ns22 = -1.0
    best_state_ns22 = None
    best_macro = -1.0
    best_state_macro = None

    val_wavs = load_val_waveforms(cfg, fold_data["val_sc_df"])

    for ep in range(cfg.epochs):
        model.train()
        sampler = MixSamp(
            list(sizes),
            list(names),
            cfg.shares,
            cfg.batch_size,
            num_steps,
            seed=cfg.seed + ep,
        )
        loader = DataLoader(
            mds,
            batch_sampler=sampler,
            collate_fn=collate_m,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        epoch_loss = epoch_cls = epoch_dist = 0.0
        num_batches = 0
        t0 = time.time()

        for wav, lb, wt, mk, src_tags in loader:
            wav = wav.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            wt = wt.to(device, non_blocking=True)
            mk = mk.to(device, non_blocking=True)
            sw = torch.tensor(
                [SOURCE_WEIGHTS.get(s, 0.0) for s in src_tags],
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                mel = mel_transform(wav)
                mean = mel.mean(dim=(2, 3), keepdim=True)
                std = mel.std(dim=(2, 3), keepdim=True) + 1e-6
                mel = (mel - mean) / std
                mel = spec_augment(mel)

            with torch.amp.autocast(device_type="cuda"):
                if cfg.use_perch_distill:
                    clip_logits, framewise, distill_emb = model(
                        mel, return_framewise=True, return_distill=True
                    )
                else:
                    clip_logits, framewise = model(mel, return_framewise=True)

                frame_max_logits = framewise.max(dim=1).values

                bce_clip = F.binary_cross_entropy_with_logits(
                    clip_logits, lb, reduction="none"
                )
                bce_frame = F.binary_cross_entropy_with_logits(
                    frame_max_logits, lb, reduction="none"
                )
                bce = 0.5 * bce_clip + 0.5 * bce_frame
                per_sample = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                cls_loss = (per_sample * sw).mean()

                if cfg.use_perch_distill and perch_teacher is not None:
                    with torch.no_grad():
                        wav_5s = wav.squeeze(1)
                        n = wav_5s.shape[1]
                        if n > cfg.val_samples:
                            start = (n - cfg.val_samples) // 2
                            wav_5s = wav_5s[:, start : start + cfg.val_samples]
                        elif n < cfg.val_samples:
                            wav_5s = F.pad(wav_5s, (0, cfg.val_samples - n))
                        perch_emb = perch_teacher.embed(wav_5s).to(device, non_blocking=True)
                    distill_loss = F.mse_loss(distill_emb, perch_emb)
                    loss = cls_loss + cfg.alpha_distill * distill_loss
                else:
                    distill_loss = torch.tensor(0.0, device=device)
                    loss = cls_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            sch.step()

            epoch_loss += loss.item()
            epoch_cls += cls_loss.item()
            epoch_dist += distill_loss.item()
            num_batches += 1

        val_preds_dict = predict_from_waveforms(
            model, mel_transform, val_wavs, device
        )
        val_preds = val_preds_dict["blend"]

        results = full_eval(
            fold_data["Y_val"], val_preds, fold_data["ns22_val"], data["taxon_masks"]
        )
        for mode in ["clip", "fmax", "blend"]:
            r_mode = full_eval(
                fold_data["Y_val"],
                val_preds_dict[mode],
                fold_data["ns22_val"],
                data["taxon_masks"],
            )
            results[f"ns22_{mode}"] = r_mode["non_s22_macro"]

        history["ep"].append(ep)
        history["train_loss"].append(epoch_loss / num_batches)
        history["cls_loss"].append(epoch_cls / num_batches)
        history["dist_loss"].append(epoch_dist / num_batches)
        history["macro"].append(results["macro_auc_all"])
        history["ns22_macro"].append(results["non_s22_macro"])
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            history[f"ns22_{t}"].append(results[f"non_s22_{t}"])
        history["val_preds"].append(val_preds)

        if results["non_s22_macro"] > best_ns22:
            best_ns22 = results["non_s22_macro"]
            best_state_ns22 = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        if results["macro_auc_all"] > best_macro:
            best_macro = results["macro_auc_all"]
            best_state_macro = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"Ep{ep:02d} "
            f"loss={epoch_loss/num_batches:.4f} "
            f"cls={epoch_cls/num_batches:.4f} "
            f"dist={epoch_dist/num_batches:.4f} "
            f"ns22={results['ns22_blend']:.4f} "
            f"[{time.time()-t0:.0f}s]"
        )

    del perch_teacher, model, mel_transform, spec_augment
    torch.cuda.empty_cache()
    gc.collect()

    return best_state_ns22, best_state_macro, history
