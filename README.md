[![Kaggle](https://img.shields.io/badge/Kaggle-BirdCLEF%202026-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/birdclef-2026)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2018-005CED?logo=onnx)](https://onnx.ai/)
[![timm](https://img.shields.io/badge/timm-EfficientNet--B0-6B2FA0)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](https://www.apache.org/licenses/LICENSE-2.0)

# BirdCLEF 2026 — Acoustic Species Classification

Multi-label bird/amphibian/insect/mammal classification from 5-second audio windows using an EfficientNet-B0 backbone with GeMFreq pooling and attention-based SED (**S**ound **E**vent **D**etection).  
Inspired by the [1st-place solution architecture](https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inference) from BirdCLEF 2025.

**Competition:** [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026)  
**Public CV:** ~0.69 macro AUC (non-S22), **234 species**

---

## Project Structure

```
├── config/
│   └── config.py              # Single dataclass for all hyperparameters
│
├── src/
│   ├── data/
│   │   ├── loader.py           # CSV loading, fold assignment, label maps, upsampling
│   │   └── dataset.py          # PyTorch datasets (FocalDS, ScDS), MixSamp sampler
│   ├── models/
│   │   ├── mel.py              # GPU Mel spectrogram + SpecAugment
│   │   ├── sed_model.py        # BirdSEDModel: EfficientNet + GeMFreq + attention
│   │   ├── perch.py            # Frozen Perch v2 teacher via ONNX
│   │   └── export.py           # ONNX export wrapper + verification
│   ├── training/
│   │   ├── train.py            # train_fold() — core training loop
│   │   ├── val.py              # Validation waveform loading + inference
│   │   └── metrics.py          # macro AUC evaluator with taxon breakdown
│   └── inference/
│       ├── preprocess.py       # Librosa mel, chunking, Gaussian smoothing
│       ├── predict.py          # Multi-fold ONNX inference engine
│       └── submission.py       # CSV writer
│
├── scripts/
│   ├── train.py                # Full training pipeline (5-fold CV + ONNX export)
│   └── predict.py              # Inference → submission.csv
│
├── runs/                       # Timestamped run directories (gitignored)
│   └── latest -> 2026-06-20_15-30-00/
│       ├── fold0_best_ns22.pt
│       ├── fold0_best_macro.pt
│       ├── sed_distill_fold0.onnx
│       ├── oof_predictions.npy
│       ├── config.json
│       └── histories.pkl
│
├── checkpoints/                # Optional: final merged/ensemble weights
├── notebooks/                  # Exploration notebooks (optional)
├── requirements.txt
└── .gitignore
```

---

## Key Features

- **GPU-native pipeline** — Mel spectrogram, SpecAugment, and normalisation all run on GPU via `torchaudio`
- **GeMFreq pooling** — Learnable generalised-mean pooling over frequency bands (sharper than mean, softer than max)
- **Attention-based SED** — Frame-level attention weights produce clip-level logits, with frame-max as a complementary head
- **5-fold cross-validation** — Stratified by species (focal) + grouped by file (soundscapes)
- **ONNX export** — Each fold exported to ONNX (opset 18) for fast CPU/GPU inference without PyTorch deps
- **Rare species upsampling** — Automatic duplication of species with fewer than `min_sample` recordings
- **Checkpoint management** — Each training run gets a timestamped directory under `runs/` with symlinked `runs/latest`

---

## Usage

### Training

```bash
# Full 5-fold training
python scripts/train.py

# Debug mode (1 fold, 1 epoch, tiny subset)
python scripts/train.py --debug

# Custom folds and epochs
python scripts/train.py --folds 0 1 2 --epochs 20 --batch-size 32

# On a local machine (CPU)
python scripts/train.py --device cpu --debug
```

Outputs are saved to `runs/<timestamp>/`:
- `fold{N}_best_ns22.pt` — best non-S22 AUC checkpoint
- `fold{N}_best_macro.pt` — best overall macro AUC checkpoint
- `sed_distill_fold{N}.onnx` — ONNX export
- `oof_predictions.npy` — out-of-fold predictions
- `config.json` — frozen config for reproducibility
- `histories.pkl` — per-epoch metrics

### Inference

```bash
# Use latest trained models
python scripts/predict.py --output submission.csv

# Specify ONNX directory and test data
python scripts/predict.py \
    --onnx-dir ./runs/latest \
    --test-dir /kaggle/input/competitions/birdclef-2026/test_soundscapes \
    --output submission.csv

# Quick debug run (first 10 files)
python scripts/predict.py --limit 10
```

---

## Config

All hyperparameters live in `config/config.py` as a `@dataclass`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `backbone_name` | `tf_efficientnet_b0.ns_jft_in1k` | TIMM backbone |
| `epochs` | 10 | Training epochs |
| `batch_size` | 16 | Batch size |
| `lr` | 5e-4 | Peak learning rate |
| `n_mels` | 256 | Mel frequency bins |
| `sr` | 32000 | Target sample rate |
| `train_duration` | 5 | Window duration (seconds) |
| `use_perch_distill` | False | Enable Perch v2 MSE distillation |
| `debug` | False | Fast mode for testing |

Override via command-line flags in `scripts/train.py` or edit `config.py` directly.

---

## Requirements

- Python 3.12+
- PyTorch 2.0+, torchaudio, torchvision
- timm 0.9+
- librosa, soundfile
- onnxruntime-gpu, onnx
- scikit-learn, scipy, pandas, numpy

See `requirements.txt` for full list.

---

## Reference

- [BirdCLEF 2026 Competition](https://www.kaggle.com/competitions/birdclef-2026)
- [1st Place Inference Notebook (2025)](https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inference)
- [Perch v2](https://www.kaggle.com/models/google/perch) — acoustic foundation model
