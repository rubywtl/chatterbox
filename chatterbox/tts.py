from __future__ import annotations

from typing import Iterator

import numpy as np
from kokoro import KPipeline

from .config import TtsConfig


class KokoroSynthesizer:
    """Streaming text-to-speech via the open-source Kokoro-82M model,
    used directly through its `KPipeline` API.
    """

    def __init__(self, config: TtsConfig):
        self._config = config
        self._pipeline = KPipeline(lang_code=config.lang_code, repo_id=config.repo_id)

    def synthesize_chunks(self, text: str) -> Iterator[np.ndarray]:
        """Yield int16 PCM chunks of `streaming_chunk_samples` for `text`."""
        generator = self._pipeline(text, voice=self._config.voice, speed=self._config.speed)
        chunk_size = self._config.streaming_chunk_samples
        carry = np.zeros(0, dtype=np.float32)

        for _graphemes, _phonemes, audio in generator:
            samples = np.concatenate([carry, np.asarray(audio, dtype=np.float32)])
            n_full = samples.shape[0] // chunk_size
            for i in range(n_full):
                piece = samples[i * chunk_size : (i + 1) * chunk_size]
                yield self._to_int16(piece)
            carry = samples[n_full * chunk_size :]

        if carry.shape[0] > 0:
            yield self._to_int16(carry)

    @staticmethod
    def _to_int16(samples: np.ndarray) -> np.ndarray:
        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
