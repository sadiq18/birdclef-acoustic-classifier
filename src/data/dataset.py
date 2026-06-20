import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler
from pathlib import Path

from config import Config


SR = 32000


def load_int16(path):
    waveform_int16 = torch.load(path, map_location="cpu")
    return waveform_int16.float() / 32767.0


_FC = {}
def load_focal(p: str, cache_dir: Path):
    if p in _FC:
        return _FC[p]
    pp = cache_dir / p
    if not pp.exists():
        return None
    a = load_int16(pp)
    if len(_FC) >= 1000:
        _FC.pop(next(iter(_FC)))
    _FC[p] = a
    return a


_SC_CACHE = {}
def load_sc_waveform(cache_dir: Path, cache_file: str):
    key = str(cache_dir / cache_file)
    if key in _SC_CACHE:
        return _SC_CACHE[key]
    pp = cache_dir / cache_file
    if not pp.exists():
        return None
    a = load_int16(pp)
    if len(_SC_CACHE) >= 100:
        _SC_CACHE.pop(next(iter(_SC_CACHE)))
    _SC_CACHE[key] = a
    return a


def extract_chunk(waveform, start_sample: int, n_samples: int):
    total = waveform.shape[-1]
    if total <= n_samples:
        pad = n_samples - total
        return F.pad(waveform, (pad, 0))
    end = start_sample + n_samples
    if end > total:
        start_sample = max(0, total - n_samples)
    return waveform[..., start_sample:start_sample + n_samples]


def apply_aug(w, cfg: Config):
    if torch.rand(1) < cfg.aug_prob:
        gain = 10 ** (torch.empty(1).uniform_(*cfg.aug_gain_db_range) / 20)
        w = w * gain
    if torch.rand(1) < cfg.aug_prob:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            noise = torch.randn_like(w)
            snr = 10 ** (torch.empty(1).uniform_(*cfg.aug_noise_snr_db_range) / 10)
            w = w + noise * torch.sqrt(sp / snr)
    return w


class FocalDS(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label2idx: dict,
        cache_dir: Path,
        train_samples: int,
        num_classes: int,
        cfg: Config,
        secondary_lookup: dict = None,
        aug: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.cache_dir = cache_dir
        self.train_samples = train_samples
        self.num_classes = num_classes
        self.cfg = cfg
        self.secondary_lookup = secondary_lookup
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def _load_chunk(self, row):
        w = load_focal(row["cache_file"], self.cache_dir)
        if w is None:
            return None, None
        if self.aug and len(w) > self.train_samples:
            start = torch.randint(0, len(w) - self.train_samples + 1, (1,)).item()
        else:
            start = int(row.get("start_sec", 0)) * SR
        ch = extract_chunk(w, start, self.train_samples)
        lb = torch.zeros(self.num_classes, dtype=torch.float32)
        label = str(row["primary_label"])
        if label in self.label2idx:
            lb[self.label2idx[label]] = 1.0
        if self.secondary_lookup is not None and "original_idx" in self.df.columns:
            for s in self.secondary_lookup.get(int(row["original_idx"]), []):
                if s in self.label2idx:
                    lb[self.label2idx[s]] = 1.0
        return ch, lb

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ch, lb = self._load_chunk(row)
        if ch is None:
            return (
                torch.zeros(1, self.train_samples),
                torch.zeros(self.num_classes),
                torch.ones(self.num_classes),
                torch.ones(self.num_classes),
                "focal_missing",
            )
        if self.aug:
            ch = apply_aug(ch, self.cfg)
        return (
            ch.unsqueeze(0).float(),
            lb.float(),
            torch.ones(self.num_classes),
            torch.ones(self.num_classes),
            "focal",
        )


class ScDS(Dataset):
    def __init__(
        self,
        Y: np.ndarray,
        sc_df: pd.DataFrame,
        cache_dir: Path,
        train_samples: int,
        num_classes: int,
        cfg: Config,
        aug: bool = False,
    ):
        self.Y = torch.from_numpy(Y).float() if isinstance(Y, np.ndarray) else Y
        self.df = sc_df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train_samples = train_samples
        self.num_classes = num_classes
        self.cfg = cfg
        self.aug = aug

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_full = None
        if row.get("cache_file"):
            wav_full = load_sc_waveform(self.cache_dir, row["cache_file"])
        if wav_full is None:
            wav_t = torch.zeros(1, self.train_samples, dtype=torch.float32)
        else:
            chunk = extract_chunk(
                wav_full, int(row["start_sec"]) * SR, self.train_samples
            )
            if self.aug:
                chunk = apply_aug(chunk, self.cfg)
            wav_t = chunk.float().unsqueeze(0)
        y = self.Y[idx]
        return (
            wav_t,
            y.float(),
            torch.ones(self.num_classes),
            torch.ones(self.num_classes),
            "sc",
        )


class MixSamp(Sampler):
    def __init__(self, sizes, names, shares, batch_size, num_steps, seed=0):
        self.sizes = sizes
        self.names = names
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)
        per_src = [
            max(1, int(round(batch_size * shares.get(n, 0.0))))
            for n in names
        ]
        diff = batch_size - sum(per_src)
        per_src[int(np.argmax(per_src))] += diff
        self.per_src = per_src
        self.offsets = torch.cumsum(torch.tensor([0] + sizes[:-1]), dim=0).tolist()

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        for _ in range(self.num_steps):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n <= 0 or size <= 0:
                    continue
                idxs = self.rng.integers(0, size, size=n)
                batch.extend((off + idxs).tolist())
            self.rng.shuffle(batch)
            yield batch


def collate_m(batch):
    wavs = torch.stack([b[0] for b in batch]).float()
    labels = torch.stack([b[1] for b in batch]).float()
    wt = torch.stack([b[2] for b in batch]).float()
    mk = torch.stack([b[3] for b in batch]).float()
    tags = [b[4] for b in batch]
    return wavs, labels, wt, mk, tags
