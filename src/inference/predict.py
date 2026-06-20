import os
import re
import glob
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional

from config import Config
from .preprocess import (
    audio_to_mel,
    file_to_chunks,
    gauss_smooth_final,
    sigmoid_inf,
    INF_CHUNK_S,
)
from src.data.loader import load_label_maps


def load_audio_32k_mono(path: str, sr: int = 32000) -> np.ndarray:
    import soundfile as sf
    import librosa
    wav, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def discover_onnx_folds(sed_dir: Path) -> list:
    pat = re.compile(r"sed_distill_fold(\d+)\.onnx$")
    folds = []
    for fname in os.listdir(sed_dir):
        m = pat.match(fname)
        if m:
            folds.append(int(m.group(1)))
    return sorted(folds)


def make_onnx_session(onnx_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


class InferenceEngine:
    def __init__(self, cfg: Config, onnx_dir: Optional[Path] = None):
        self.cfg = cfg
        self.primary_labels, self.label2idx, self.sample_sub = load_label_maps(cfg)
        self.num_classes = len(self.primary_labels)

        onnx_dir = onnx_dir or cfg.out_path
        inf_folds = discover_onnx_folds(onnx_dir)
        if not inf_folds:
            raise FileNotFoundError(f"No ONNX files found in {onnx_dir}")

        self.fold_sessions = []
        for fold in inf_folds:
            p = onnx_dir / f"sed_distill_fold{fold}.onnx"
            sess = make_onnx_session(str(p))
            self.fold_sessions.append(sess)

        self.num_folds = len(self.fold_sessions)

    def predict_file(self, file_path: str) -> np.ndarray:
        wav = load_audio_32k_mono(file_path)
        chunks, _ = file_to_chunks(wav)
        mel = audio_to_mel(chunks)

        logits_sum = np.zeros((chunks.shape[0], self.num_classes), dtype=np.float32)
        for sess in self.fold_sessions:
            outs = sess.run(None, {"mel": mel})
            clip_logits = outs[0]
            frame_max = outs[1].max(axis=1)
            logits_sum += 0.5 * clip_logits + 0.5 * frame_max

        logits_mean = logits_sum / self.num_folds
        logits_smoothed = gauss_smooth_final(logits_mean, self.num_classes)
        probs = sigmoid_inf(logits_smoothed)
        return probs

    def predict_all(
        self, test_dir: Path, limit: Optional[int] = None
    ) -> tuple:
        test_files = sorted(glob.glob(f"{test_dir}/*.ogg"))
        if not test_files:
            fallback = self.cfg.comp_path / "train_soundscapes"
            if fallback.is_dir():
                test_files = sorted(glob.glob(f"{fallback}/*.ogg"))[:5]

        if limit:
            test_files = test_files[:limit]

        t0 = time.time()
        all_rows, all_preds = [], []

        for file_idx, file_path in enumerate(test_files):
            basename = os.path.basename(file_path).replace(".ogg", "")
            _, end_times = file_to_chunks(np.zeros(60 * INF_CHUNK_S))
            probs = self.predict_file(file_path)
            all_rows.extend([f"{basename}_{int(t)}" for t in end_times])
            all_preds.append(probs)

            if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == len(test_files) - 1:
                elapsed = time.time() - t0
                rate = (file_idx + 1) / elapsed
                print(f"  [{file_idx+1:4d}/{len(test_files)}] {elapsed:.1f}s  {rate:.2f} files/s")

        all_preds_arr = np.concatenate(all_preds) if all_preds else np.zeros((0, self.num_classes), np.float32)
        return all_rows, all_preds_arr
