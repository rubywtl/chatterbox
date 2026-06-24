from __future__ import annotations

import webrtcvad
import numpy as np

from .config import AudioConfig, VadConfig


class FrameVad:
    """Thin wrapper around webrtcvad for per-frame speech/silence classification.

    This is the cheap, always-on gate that decides *whether* someone is
    talking. The Smart Turn model (turn_taking.py) is the more expensive,
    semantic layer that decides *whether they're done*.
    """

    def __init__(self, audio_config: AudioConfig, vad_config: VadConfig):
        if audio_config.frame_ms not in (10, 20, 30):
            raise ValueError("webrtcvad requires frame_ms to be 10, 20, or 30")
        self._vad = webrtcvad.Vad(vad_config.aggressiveness)
        self._sample_rate = audio_config.input_sample_rate

    def is_speech(self, frame: np.ndarray) -> bool:
        return self._vad.is_speech(frame.tobytes(), self._sample_rate)
