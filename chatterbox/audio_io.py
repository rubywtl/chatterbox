from __future__ import annotations

import asyncio
import queue
import threading

import numpy as np

from .config import AudioConfig

# sounddevice (PortAudio) is only imported lazily, inside start(), so that
# importing this module — which orchestrator.py always does — doesn't
# require PortAudio to be installed on machines that never touch local
# audio, e.g. a headless server using the network-backed streams instead.


class MicStream:
    """Captures mic audio and exposes it as an async iterator of int16 frames.

    Each frame is exactly `config.frame_samples` samples, which is what
    webrtcvad and the rest of the pipeline assume.
    """

    def __init__(self, config: AudioConfig):
        self._config = config
        self._frame_samples = config.frame_samples
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            pass  # overflow/underflow on the input device; frame is still usable
        frame = indata[:, 0].copy()
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)

    def start(self) -> None:
        import sounddevice as sd

        self._loop = asyncio.get_event_loop()
        self._stream = sd.InputStream(
            samplerate=self._config.input_sample_rate,
            channels=self._config.input_channels,
            dtype="int16",
            blocksize=self._frame_samples,
            device=self._config.input_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def frames(self):
        while True:
            yield await self._queue.get()


class SpeakerStream:
    """Plays back int16 PCM chunks pushed from anywhere, with instant `clear()`
    for barge-in interrupts.
    """

    def __init__(self, config: AudioConfig):
        self._config = config
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._leftover = np.zeros(0, dtype=np.int16)
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self.is_playing = False

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            buf = self._leftover
            while buf.shape[0] < frames:
                try:
                    chunk = self._queue.get_nowait()
                except queue.Empty:
                    break
                buf = np.concatenate([buf, chunk])
            if buf.shape[0] >= frames:
                out = buf[:frames]
                self._leftover = buf[frames:]
            else:
                out = np.concatenate([buf, np.zeros(frames - buf.shape[0], dtype=np.int16)])
                self._leftover = np.zeros(0, dtype=np.int16)
            self.is_playing = self._leftover.shape[0] > 0 or not self._queue.empty()
        outdata[:, 0] = out

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self._config.output_sample_rate,
            channels=self._config.output_channels,
            dtype="int16",
            blocksize=self._config.output_chunk_frames,
            device=self._config.output_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def write(self, chunk: np.ndarray) -> None:
        self._queue.put(chunk)
        with self._lock:
            self.is_playing = True

    async def clear(self) -> None:
        """Drop all queued/buffered audio immediately (barge-in)."""
        with self._lock:
            self._leftover = np.zeros(0, dtype=np.int16)
            self.is_playing = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def drained(self) -> bool:
        with self._lock:
            return self._leftover.shape[0] == 0 and self._queue.empty()

    async def send_transcript(self, role: str, text: str) -> None:
        pass  # no display surface for local mic/speaker mode
