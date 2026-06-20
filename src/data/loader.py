import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold

from config import Config


def load_label_maps(cfg: Config):
    sample_sub = pd.read_csv(cfg.sample_sub_path)
    primary_labels = sample_sub.columns[1:].tolist()
    label2idx = {lbl: i for i, lbl in enumerate(primary_labels)}
    return primary_labels, label2idx, sample_sub


def load_taxonomy(cfg: Config, primary_labels: list[str]):
    taxonomy = pd.read_csv(cfg.taxonomy_path)
    label_to_taxon = dict(
        zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str))
    )
    taxon_masks = {}
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        taxon_masks[t] = np.array([
            i for i, l in enumerate(primary_labels)
            if label_to_taxon.get(l, "") == t
        ])
    return taxonomy, label_to_taxon, taxon_masks


def load_focal_data(cfg: Config, label2idx: dict[str, int]):
    audio_cache_meta = pd.read_csv(cfg.waveform_cache_path / "audio_cache_meta.csv")
    train_df = pd.read_csv(cfg.train_csv_path)
    audio_cache_meta = audio_cache_meta.merge(
        train_df[["filename", "secondary_labels"]], on="filename", how="left"
    )
    audio_cache_meta = audio_cache_meta[
        audio_cache_meta["primary_label"].isin(label2idx)
    ].reset_index(drop=True)
    return audio_cache_meta


def build_soundscape_labels(cfg: Config, label2idx: dict[str, int]):
    sc_labels_raw = pd.read_csv(cfg.labels_path).drop_duplicates()
    sc_labels_raw["start_sec"] = (
        pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    )
    sc_cache_meta = pd.read_csv(cfg.waveform_cache_path / "soundscape_cache_meta.csv")
    sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
        lambda x: x.split(";") if isinstance(x, str) else []
    )

    Y_SC = np.zeros((len(sc_cache_meta), cfg.num_classes), dtype=np.float32)
    for i, row in sc_cache_meta.iterrows():
        matches = sc_labels_raw[
            (sc_labels_raw["filename"] == row["filename"])
            & (sc_labels_raw["start_sec"] == row["start_sec"])
        ]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in label2idx:
                    Y_SC[i, label2idx[lbl]] = 1.0

    labeled_sc_mask = Y_SC.sum(axis=1) > 0
    non_s22_mask_sc = sc_cache_meta["site"].values != "S22"
    return sc_cache_meta, Y_SC, non_s22_mask_sc, labeled_sc_mask


def assign_focal_folds(cfg: Config, audio_cache_meta: pd.DataFrame, label2idx: dict[str, int]):
    audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    audio_for_split["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        skf.split(audio_for_split, audio_for_split["primary_label"])
    ):
        audio_for_split.loc[val_idx, "fold"] = fold
    audio_cache_meta = audio_cache_meta.merge(
        audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
    )
    return audio_cache_meta


def assign_sc_folds(cfg: Config, sc_cache_meta: pd.DataFrame):
    sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=cfg.n_folds)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_cache_meta["fold"] = (
        sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    )
    return sc_cache_meta


def upsample_rare(cfg: Config, audio_cache_meta: pd.DataFrame):
    counts = audio_cache_meta["primary_label"].value_counts()
    rare_species = counts[counts < cfg.min_sample].index
    extra_rows = []
    for sp in rare_species:
        sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
        n_copies = int(np.ceil(cfg.min_sample / len(sp_rows))) - 1
        for _ in range(n_copies):
            extra_rows.append(sp_rows)
    n_before = len(audio_cache_meta)
    if extra_rows:
        audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows, ignore_index=True)
    return audio_cache_meta, n_before


def load_secondary_labels(cfg: Config, label2idx: dict[str, int]):
    train_df = pd.read_csv(cfg.train_csv_path)
    secondary = {}
    for idx, row in train_df.iterrows():
        sec = row.get("secondary_labels", "")
        if pd.isna(sec) or sec in ("", "[]"):
            continue
        try:
            sec_list = eval(sec) if isinstance(sec, str) else []
        except Exception:
            continue
        valid = [s for s in sec_list if s in label2idx]
        if valid:
            secondary[idx] = valid
    return secondary


def build_sc_mixup_pool(cfg: Config, sc_cache_meta: pd.DataFrame, Y_SC: np.ndarray):
    sc_file_meta = pd.read_csv(cfg.waveform_cache_path / "soundscape_file_meta.csv")
    sc_file_dict = dict(zip(sc_file_meta["filename"], sc_file_meta["cache_file"]))
    labeled_rows = []
    for i in range(len(sc_cache_meta)):
        row = sc_cache_meta.iloc[i]
        if Y_SC[i].sum() > 0:
            cf = sc_file_dict.get(row["filename"])
            if cf is not None:
                labeled_rows.append({
                    "filename": row["filename"],
                    "start_sec": int(row["start_sec"]),
                    "cache_file": cf,
                    "label_idx": i,
                    "fold": int(row.get("fold", -1)),
                })
    if labeled_rows:
        labeled_meta = pd.DataFrame(labeled_rows)
        return [(cfg.waveform_cache_path, labeled_meta, Y_SC)]
    return []


def load_data(cfg: Config):
    primary_labels, label2idx, sample_sub = load_label_maps(cfg)
    taxonomy, label_to_taxon, taxon_masks = load_taxonomy(cfg, primary_labels)
    audio_cache_meta = load_focal_data(cfg, label2idx)
    sc_cache_meta, Y_SC, non_s22_mask_sc, labeled_sc_mask = build_soundscape_labels(cfg, label2idx)

    audio_cache_meta = assign_focal_folds(cfg, audio_cache_meta, label2idx)
    sc_cache_meta = assign_sc_folds(cfg, sc_cache_meta)

    audio_cache_meta, n_before = upsample_rare(cfg, audio_cache_meta)

    secondary_labels = load_secondary_labels(cfg, label2idx) if cfg.use_focal_secondary else None
    sc_mixup_pool = build_sc_mixup_pool(cfg, sc_cache_meta, Y_SC)

    data = {
        "primary_labels": primary_labels,
        "label2idx": label2idx,
        "sample_sub": sample_sub,
        "taxonomy": taxonomy,
        "label_to_taxon": label_to_taxon,
        "taxon_masks": taxon_masks,
        "audio_cache_meta": audio_cache_meta,
        "sc_cache_meta": sc_cache_meta,
        "Y_SC": Y_SC,
        "non_s22_mask_sc": non_s22_mask_sc,
        "labeled_sc_mask": labeled_sc_mask,
        "secondary_labels": secondary_labels,
        "sc_mixup_pool": sc_mixup_pool,
        "n_focal_before_upsample": n_before,
    }
    return data


def get_fold_data(data: dict, fold_k: int):
    train_sc_mask = data["sc_cache_meta"]["fold"].values != fold_k
    val_sc_mask = data["sc_cache_meta"]["fold"].values == fold_k
    return {
        "train_sc_mask": train_sc_mask,
        "val_sc_mask": val_sc_mask,
        "Y_val": data["Y_SC"][val_sc_mask],
        "ns22_val": data["non_s22_mask_sc"][val_sc_mask],
        "val_sc_df": data["sc_cache_meta"][val_sc_mask].reset_index(drop=True),
    }
