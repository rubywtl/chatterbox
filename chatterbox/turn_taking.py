from __future__ import annotations

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

from .config import TurnTakingConfig


def _pad_or_truncate_to_end(audio: np.ndarray, n_seconds: int, sample_rate: int) -> np.ndarray:
    """Keep the most recent n_seconds of audio; if shorter, zero-pad at the
    front so the real audio sits at the end of the window. Mirrors the
    reference preprocessing in pipecat-ai/smart-turn's audio_utils.py.
    """
    max_samples = n_seconds * sample_rate
    if audio.shape[0] > max_samples:
        return audio[-max_samples:]
    if audio.shape[0] < max_samples:
        pad = max_samples - audio.shape[0]
        return np.pad(audio, (pad, 0), mode="constant", constant_values=0)
    return audio


class SmartTurnDetector:
    """End-of-turn classifier using pipecat-ai's Smart Turn v3 ONNX model.

    Loaded and run directly via onnxruntime + a Whisper feature extractor —
    no pipecat runtime involved, just the open model weights.
    """

    SAMPLE_RATE = 16000

    def __init__(self, config: TurnTakingConfig):
        self._config = config
        model_path = hf_hub_download(repo_id=config.model_repo, filename=config.model_file)

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if config.device == "cuda" else ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        self._feature_extractor = WhisperFeatureExtractor(chunk_length=config.window_seconds)

    def probability_complete(self, audio_int16: np.ndarray) -> float:
        """Return the probability that the given trailing audio window
        represents a completed conversational turn.
        """
        audio = audio_int16.astype(np.float32) / 32768.0
        audio = _pad_or_truncate_to_end(audio, self._config.window_seconds, self.SAMPLE_RATE)

        max_length = self._config.window_seconds * self.SAMPLE_RATE
        inputs = self._feature_extractor(
            audio,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        outputs = self._session.run(None, {"input_features": input_features})
        return float(outputs[0][0].item())

    def is_complete(self, audio_int16: np.ndarray) -> bool:
        return self.probability_complete(audio_int16) > self._config.endpoint_threshold
