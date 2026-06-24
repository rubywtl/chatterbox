# chatterbox

A local, full-duplex speech agent built from scratch in Python — no agent
framework dependency. Every model is called directly:

- **Turn-taking**: [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) (open-source ONNX model, run via `onnxruntime`) decides when the user has actually finished speaking, layered on top of a fast frame-level VAD (`webrtcvad`).
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (local, open-source).
- **LLM**: [Ollama](https://ollama.com) served locally, called over its native HTTP streaming API.
- **TTS**: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (open-source), streamed in configurable chunk sizes.

The whole pipeline — mic capture, VAD, end-of-turn detection, STT, streaming
LLM, streaming TTS, playback, and barge-in — is custom asyncio code in
[chatterbox/](chatterbox/). [config.yaml](config.yaml) is the single place
that lists model names, devices, and every streaming/buffering size (mic
frame size, Smart Turn window, LLM sentence-chunk size, TTS streaming chunk
size, playback buffer size).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ollama must be running locally with the model pulled:
ollama pull llama3.1:8b
ollama serve  # if not already running
```

Smart Turn and Kokoro weights are downloaded automatically from the Hugging
Face Hub on first run.

## Run

```bash
python -m chatterbox.main --config config.yaml
```

Talk into your mic; the agent replies out loud and you can interrupt it
mid-sentence (barge-in) by just starting to talk again.

## Tuning

All of this lives in [config.yaml](config.yaml):

- `audio.frame_ms` / `output_chunk_frames` — mic/playback buffer sizes (latency vs. underrun tradeoff).
- `vad.*` — how fast we detect speech start and how long we wait in silence before checking for end-of-turn.
- `turn_taking.*` — which Smart Turn ONNX checkpoint to use (`cpu` vs `gpu` variant) and its decision threshold.
- `stt.*` — Whisper model size/device/precision.
- `llm.*` — Ollama model name, sampling params, and `sentence_chunk_chars` (how much text to buffer before sending a sentence to TTS).
- `tts.*` — Kokoro voice/speed and `streaming_chunk_samples` (audio chunk size pushed to the speaker).
