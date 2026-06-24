from __future__ import annotations

import json
import time

import numpy as np

from .config import AudioConfig

_PCM_DTYPE = np.dtype("<i2")  # explicit little-endian int16


def encode_pcm(frame: np.ndarray) -> bytes:
    return frame.astype(_PCM_DTYPE, copy=False).tobytes()


def decode_pcm(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=_PCM_DTYPE)


class WebSocketMicStream:
    """Drop-in replacement for MicStream backed by a browser tab's mic,
    delivered as binary WebSocket messages (one int16 PCM frame each).
    """

    def __init__(self, websocket):
        self._ws = websocket

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def frames(self):
        async for message in self._ws:
            if isinstance(message, bytes):
                yield decode_pcm(message)
            # text messages from the client aren't used today; ignored.


class WebSocketSpeakerStream:
    """Drop-in replacement for SpeakerStream that ships audio to a browser
    tab over a WebSocket: binary messages are PCM chunks, text messages are
    small JSON control/info events (`clear`, `transcript`).

    `drained()` uses the same virtual playback clock trick as
    NetworkSpeakerStream, since the server can't observe the browser's
    actual audio buffer.
    """

    def __init__(self, websocket, audio_config: AudioConfig):
        self._ws = websocket
        self._sample_rate = audio_config.output_sample_rate
        self._play_until = 0.0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def write(self, chunk: np.ndarray) -> None:
        await self._ws.send(encode_pcm(chunk))
        now = time.monotonic()
        duration = chunk.shape[0] / self._sample_rate
        self._play_until = max(self._play_until, now) + duration

    async def clear(self) -> None:
        await self._ws.send(json.dumps({"type": "clear"}))
        self._play_until = 0.0

    def drained(self) -> bool:
        return time.monotonic() >= self._play_until

    async def send_transcript(self, role: str, text: str) -> None:
        await self._ws.send(json.dumps({"type": "transcript", "role": role, "text": text}))
