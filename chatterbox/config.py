from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    input_sample_rate: int = 16000
    input_channels: int = 1
    frame_ms: int = 30
    output_sample_rate: int = 24000
    output_channels: int = 1
    output_chunk_frames: int = 1024
    input_device: str | int | None = None
    output_device: str | int | None = None

    @property
    def frame_samples(self) -> int:
        return self.input_sample_rate * self.frame_ms // 1000


@dataclass
class VadConfig:
    backend: str = "webrtcvad"
    aggressiveness: int = 2
    speech_start_frames: int = 3
    silence_hold_ms: int = 400
    hard_timeout_ms: int = 3000
    recheck_ms: int = 200


@dataclass
class TurnTakingConfig:
    model_repo: str = "pipecat-ai/smart-turn-v3"
    model_file: str = "smart-turn-v3.2-cpu.onnx"
    window_seconds: int = 8
    endpoint_threshold: float = 0.5
    device: str = "cpu"


@dataclass
class SttConfig:
    engine: str = "faster-whisper"
    model_size: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1


@dataclass
class LlmConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    system_prompt: str = "You are a concise, friendly voice assistant."
    temperature: float = 0.7
    max_tokens: int = 512
    request_timeout_s: float = 60.0
    sentence_chunk_chars: int = 40
    sentence_boundary_chars: str = ".?!\n"


@dataclass
class TtsConfig:
    engine: str = "kokoro"
    repo_id: str = "hexgrad/Kokoro-82M"
    lang_code: str = "a"
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000
    streaming_chunk_samples: int = 4096


@dataclass
class BargeInConfig:
    enabled: bool = True
    min_speech_frames_to_interrupt: int = 3


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    turn_taking: TurnTakingConfig = field(default_factory=TurnTakingConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            audio=AudioConfig(**raw.get("audio", {})),
            vad=VadConfig(**raw.get("vad", {})),
            turn_taking=TurnTakingConfig(**raw.get("turn_taking", {})),
            stt=SttConfig(**raw.get("stt", {})),
            llm=LlmConfig(**raw.get("llm", {})),
            tts=TtsConfig(**raw.get("tts", {})),
            barge_in=BargeInConfig(**raw.get("barge_in", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )
