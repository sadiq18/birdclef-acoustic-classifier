import torch
import torch.nn as nn
import torchaudio

from config import Config


class MelSpecTransform(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


class SpecAugment(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=cfg.freq_mask_param
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=cfg.time_mask_param
        )
        self.num_freq_masks = cfg.num_freq_masks
        self.num_time_masks = cfg.num_time_masks

    def forward(self, mel):
        for _ in range(self.num_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.num_time_masks):
            mel = self.time_mask(mel)
        return mel
