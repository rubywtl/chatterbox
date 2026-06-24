from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from .config import SttConfig


class WhisperTranscriber:
    """Local speech-to-text via faster-whisper (CTranslate2 backend)."""

    def __init__(self, config: SttConfig):
        self._config = config
        self._model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )

    def transcribe(self, audio_int16: np.ndarray) -> str:
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language=self._config.language,
            beam_size=self._config.beam_size,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
