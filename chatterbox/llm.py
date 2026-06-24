from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .config import LlmConfig


class OllamaClient:
    """Streams chat completions from a local Ollama server's native API
    (POST /api/chat, newline-delimited JSON), and re-chunks the token
    stream into sentence-sized pieces ready to hand to TTS.
    """

    def __init__(self, config: LlmConfig):
        self._config = config
        self._history: list[dict[str, str]] = [{"role": "system", "content": config.system_prompt}]

    async def stream_tokens(self, user_text: str) -> AsyncIterator[str]:
        self._history.append({"role": "user", "content": user_text})
        payload = {
            "model": self._config.model,
            "messages": self._history,
            "stream": True,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }

        reply_parts: list[str] = []
        timeout = httpx.Timeout(self._config.request_timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self._config.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        reply_parts.append(token)
                        yield token
                    if data.get("done"):
                        break

        self._history.append({"role": "assistant", "content": "".join(reply_parts)})

    async def stream_sentences(self, user_text: str) -> AsyncIterator[str]:
        """Re-chunk the raw token stream into sentence-ish pieces so TTS can
        start speaking before the LLM has finished the full reply.
        """
        buf = ""
        async for token in self.stream_tokens(user_text):
            buf += token
            if len(buf) >= self._config.sentence_chunk_chars and any(
                c in self._config.sentence_boundary_chars for c in buf
            ):
                boundary = max(buf.rfind(c) for c in self._config.sentence_boundary_chars if c in buf)
                chunk, buf = buf[: boundary + 1], buf[boundary + 1 :]
                chunk = chunk.strip()
                if chunk:
                    yield chunk
        buf = buf.strip()
        if buf:
            yield buf
