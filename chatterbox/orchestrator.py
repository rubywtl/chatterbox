from __future__ import annotations

import asyncio
import enum
import logging
import queue as sync_queue
import threading
from typing import AsyncIterator, Callable, Iterator

import numpy as np

from .audio_io import MicStream, SpeakerStream
from .config import Config
from .llm import OllamaClient
from .stt import WhisperTranscriber
from .tts import KokoroSynthesizer
from .turn_taking import SmartTurnDetector
from .vad import FrameVad

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    PROCESSING = "processing"
    AGENT_SPEAKING = "agent_speaking"


async def _iter_in_thread(sync_gen_factory: Callable[[], Iterator]) -> AsyncIterator:
    """Bridge a blocking (CPU-bound) sync generator into an async iterator
    by running it on a worker thread, so the event loop stays free to keep
    reading mic frames (needed for barge-in) while it runs.
    """
    q: "sync_queue.Queue" = sync_queue.Queue(maxsize=8)
    sentinel = object()

    def worker():
        try:
            for item in sync_gen_factory():
                q.put(item)
        except Exception as exc:  # surfaced to the consumer below
            q.put(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await asyncio.to_thread(q.get)
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


class Agent:
    """Custom asyncio speech-agent pipeline: mic -> VAD -> Smart Turn ->
    Whisper STT -> Ollama LLM -> Kokoro TTS -> speaker, with barge-in.

    No external agent framework involved — every model is called directly.
    """

    def __init__(
        self,
        config: Config,
        mic=None,
        speaker=None,
        turn_detector: SmartTurnDetector | None = None,
        stt: WhisperTranscriber | None = None,
        tts: KokoroSynthesizer | None = None,
    ):
        """`mic`/`speaker`/`turn_detector`/`stt`/`tts` are injectable so a
        server process can load the heavy models once and reuse them across
        sessions, and swap local audio I/O for network-backed I/O (see
        web_audio.py) when the mic/speaker live on a remote browser tab.
        """
        self.config = config
        self.mic = mic if mic is not None else MicStream(config.audio)
        self.speaker = speaker if speaker is not None else SpeakerStream(config.audio)
        self.frame_vad = FrameVad(config.audio, config.vad)

        if turn_detector is not None:
            self.turn_detector = turn_detector
        else:
            logger.info("Loading Smart Turn v3 (%s)...", config.turn_taking.model_file)
            self.turn_detector = SmartTurnDetector(config.turn_taking)

        if stt is not None:
            self.stt = stt
        else:
            logger.info("Loading faster-whisper (%s)...", config.stt.model_size)
            self.stt = WhisperTranscriber(config.stt)

        self.llm = OllamaClient(config.llm)  # per-session chat history, always fresh

        if tts is not None:
            self.tts = tts
        else:
            logger.info("Loading Kokoro (%s)...", config.tts.repo_id)
            self.tts = KokoroSynthesizer(config.tts)

        self.state = State.IDLE
        self._utterance_frames: list[np.ndarray] = []
        self._silence_ms = 0
        self._since_last_check_ms = 0
        self._speech_run = 0
        self._response_task: asyncio.Task | None = None

    async def run(self) -> None:
        self.mic.start()
        self.speaker.start()
        logger.info("Agent ready. Listening...")
        try:
            async for frame in self.mic.frames():
                await self._handle_frame(frame)
        finally:
            self.mic.stop()
            self.speaker.stop()

    def _begin_user_turn(self, frame: np.ndarray) -> None:
        self.state = State.USER_SPEAKING
        self._utterance_frames = [frame]
        self._silence_ms = 0
        self._since_last_check_ms = 0
        self._speech_run = 0

    async def _interrupt(self) -> None:
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
        await self.speaker.clear()

    async def _handle_frame(self, frame: np.ndarray) -> None:
        is_speech = self.frame_vad.is_speech(frame)

        if self.state == State.AGENT_SPEAKING:
            if self.config.barge_in.enabled and is_speech:
                self._speech_run += 1
                if self._speech_run >= self.config.barge_in.min_speech_frames_to_interrupt:
                    await self._interrupt()
                    self._begin_user_turn(frame)
                return
            self._speech_run = 0
            task_done = self._response_task is None or self._response_task.done()
            if self.speaker.drained() and task_done:
                self.state = State.IDLE
            return

        if self.state == State.PROCESSING:
            return

        if self.state == State.IDLE:
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.config.vad.speech_start_frames:
                    self._begin_user_turn(frame)
            else:
                self._speech_run = 0
            return

        # state == USER_SPEAKING
        self._utterance_frames.append(frame)
        frame_ms = self.config.audio.frame_ms
        if is_speech:
            self._silence_ms = 0
            self._since_last_check_ms = 0
            return

        self._silence_ms += frame_ms
        self._since_last_check_ms += frame_ms
        if self._silence_ms < self.config.vad.silence_hold_ms:
            return

        hard_timeout = self._silence_ms >= self.config.vad.hard_timeout_ms
        if not hard_timeout and self._since_last_check_ms < self.config.vad.recheck_ms:
            return

        self._since_last_check_ms = 0
        audio = np.concatenate(self._utterance_frames)
        turn_complete = hard_timeout or self.turn_detector.is_complete(audio)
        if turn_complete:
            self._finalize_user_turn(audio)

    def _finalize_user_turn(self, audio: np.ndarray) -> None:
        self.state = State.PROCESSING
        self._utterance_frames = []
        self._silence_ms = 0
        self._since_last_check_ms = 0
        self._speech_run = 0
        self._response_task = asyncio.create_task(self._respond(audio))

    async def _respond(self, audio: np.ndarray) -> None:
        try:
            text = (await asyncio.to_thread(self.stt.transcribe, audio)).strip()
            if not text:
                self.state = State.IDLE
                return
            logger.info("user: %s", text)
            await self.speaker.send_transcript("user", text)

            spoke_any = False
            async for sentence in self.llm.stream_sentences(text):
                logger.info("agent: %s", sentence)
                await self.speaker.send_transcript("agent", sentence)
                async for chunk in _iter_in_thread(lambda s=sentence: self.tts.synthesize_chunks(s)):
                    spoke_any = True
                    self.state = State.AGENT_SPEAKING
                    await self.speaker.write(chunk)

            if not spoke_any:
                self.state = State.IDLE
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("response pipeline failed")
            self.state = State.IDLE
