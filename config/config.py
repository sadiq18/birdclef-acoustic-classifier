from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class Config:
    seed: int = 42
    mode: str = "train"
    debug: bool = False

    comp_dir: str = "/kaggle/input/competitions/birdclef-2026"
    waveform_cache_dir: str = "/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache"
    perch_onnx_path: str = "/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx"
    out_dir: str = "./runs"

    sr: int = 32000
    train_duration: int = 5
    val_duration: int = 5

    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 256
    f_min: int = 20
    f_max: int = 16000

    backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k"
    num_classes: int = 234
    hidden_dim: int = 512
    drop_path_rate: float = 0.1

    use_perch_distill: bool = False
    perch_embed_dim: int = 1536
    alpha_distill: float = 1.0

    n_folds: int = 5
    folds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    epochs: int = 10
    batch_size: int = 16
    lr: float = 5e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip_norm: float = 1.0

    min_sample: int = 20

    aug_prob: float = 0.5
    aug_gain_db_range: tuple = (-6.0, 6.0)
    aug_noise_snr_db_range: tuple = (10.0, 30.0)
    aug_shift_samples_max: int = 16000

    use_focal_mixup: bool = True
    mixup_prob: float = 0.5
    mixup_alpha: float = 0.4
    mixup_hard: bool = True

    use_focal_sc_mixup: bool = True
    focal_sc_mixup_prob: float = 0.5
    focal_sc_mixup_alpha: float = 0.4

    freq_mixstyle_prob: float = 0.0
    freq_mixstyle_alpha: float = 0.1

    freq_mask_param: int = 10
    time_mask_param: int = 10
    num_freq_masks: int = 1
    num_time_masks: int = 2

    use_focal: bool = True
    use_focal_secondary: bool = False
    use_labeled_sc: bool = False

    active_sources: list = field(default_factory=lambda: ["focal", "sc"])
    shares: dict = field(default_factory=lambda: {"focal": 0.9, "sc": 0.1})
    source_weights: dict = field(default_factory=lambda: {"focal": 1.0, "focal_missing": 0.0, "sc": 1.0})

    @property
    def comp_path(self) -> Path:
        return Path(self.comp_dir)

    @property
    def waveform_cache_path(self) -> Path:
        return Path(self.waveform_cache_dir)

    @property
    def perch_onnx(self) -> Path:
        return Path(self.perch_onnx_path)

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)

    @property
    def labels_path(self) -> Path:
        return self.comp_path / "train_soundscapes_labels.csv"

    @property
    def taxonomy_path(self) -> Path:
        return self.comp_path / "taxonomy.csv"

    @property
    def sample_sub_path(self) -> Path:
        return self.comp_path / "sample_submission.csv"

    @property
    def test_dir(self) -> Path:
        return self.comp_path / "test_soundscapes"

    @property
    def train_csv_path(self) -> Path:
        return self.comp_path / "train.csv"

    @property
    def train_samples(self) -> int:
        return self.sr * self.train_duration

    @property
    def val_samples(self) -> int:
        return self.sr * self.val_duration
