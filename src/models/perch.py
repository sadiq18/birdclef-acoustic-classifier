import torch
import onnxruntime as ort
from pathlib import Path

from config import Config


class PerchTeacher:
    def __init__(self, onnx_path: Path, device_str: str = "cuda"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device_str == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._out_names = [o.name for o in self.session.get_outputs()]
        self._embed_idx = None
        for i, o in enumerate(self.session.get_outputs()):
            if o.shape and o.shape[-1] == 1536:
                self._embed_idx = i
                break
        if self._embed_idx is None:
            self._embed_idx = 1

    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s
        results = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(results[self._embed_idx]).float()
