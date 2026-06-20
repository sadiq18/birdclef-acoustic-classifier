import numpy as np
import librosa
from scipy.ndimage import convolve1d

INF_SR = 32000
INF_N_MELS = 256
INF_N_FFT = 2048
INF_HOP = 512
INF_FMIN = 20
INF_FMAX = 16000
INF_TOP_DB = 80
INF_CHUNK_S = 5
INF_CHUNK_N = INF_SR * INF_CHUNK_S

GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
N_WINDOWS = 12


def audio_to_mel(chunks: np.ndarray) -> np.ndarray:
    mels = []
    for i in range(chunks.shape[0]):
        S = librosa.feature.melspectrogram(
            y=chunks[i],
            sr=INF_SR,
            n_fft=INF_N_FFT,
            hop_length=INF_HOP,
            n_mels=INF_N_MELS,
            fmin=INF_FMIN,
            fmax=INF_FMAX,
            power=2.0,
        )
        S_dB = librosa.power_to_db(S, top_db=INF_TOP_DB)
        S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
        mels.append(S_dB)
    return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)


def file_to_chunks(wav: np.ndarray, target_len: int = 60 * INF_SR) -> tuple:
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    elif len(wav) > target_len:
        wav = wav[:target_len]
    n_chunks = target_len // INF_CHUNK_N
    chunks = wav[: n_chunks * INF_CHUNK_N].reshape(n_chunks, INF_CHUNK_N)
    end_times = np.arange(1, n_chunks + 1) * INF_CHUNK_S
    return chunks.astype(np.float32), end_times


def sigmoid_inf(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
        np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
    ).astype(np.float32)


def gauss_smooth_final(
    scores: np.ndarray, num_classes: int = 234
) -> np.ndarray:
    smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], GAUSSIAN_KERNEL, axis=0, mode="nearest")
    return smoothed.reshape(-1, scores.shape[1])
