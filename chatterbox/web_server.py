from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import websockets

from .config import Config
from .orchestrator import Agent
from .stt import WhisperTranscriber
from .tts import KokoroSynthesizer
from .turn_taking import SmartTurnDetector
from .web_audio import WebSocketMicStream, WebSocketSpeakerStream

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "web"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="chatterbox browser-client server (run this on the DGX)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (keep this on localhost and use an SSH tunnel)")
    parser.add_argument("--http-port", type=int, default=8766, help="Serves the static page")
    parser.add_argument("--ws-port", type=int, default=8765, help="Audio WebSocket bridge")
    return parser.parse_args()


def _serve_static(host: str, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(STATIC_DIR))
    ThreadingHTTPServer((host, port), handler).serve_forever()


class WebServer:
    """Loads every model once and serves one speech-agent session per
    browser tab that connects to the audio WebSocket.
    """

    def __init__(self, config: Config):
        self.config = config
        logger.info("Loading Smart Turn v3 (%s)...", config.turn_taking.model_file)
        self.turn_detector = SmartTurnDetector(config.turn_taking)
        logger.info("Loading faster-whisper (%s)...", config.stt.model_size)
        self.stt = WhisperTranscriber(config.stt)
        logger.info("Loading Kokoro (%s)...", config.tts.repo_id)
        self.tts = KokoroSynthesizer(config.tts)

    async def handle_connection(self, websocket) -> None:
        logger.info("browser client connected: %s", websocket.remote_address)

        handshake = {
            "input_sample_rate": self.config.audio.input_sample_rate,
            "input_channels": self.config.audio.input_channels,
            "frame_ms": self.config.audio.frame_ms,
            "output_sample_rate": self.config.audio.output_sample_rate,
        }
        await websocket.send(json.dumps(handshake))

        agent = Agent(
            self.config,
            mic=WebSocketMicStream(websocket),
            speaker=WebSocketSpeakerStream(websocket, self.config.audio),
            turn_detector=self.turn_detector,
            stt=self.stt,
            tts=self.tts,
        )
        try:
            await agent.run()
        except websockets.ConnectionClosed:
            logger.info("browser client disconnected")
        except Exception:
            logger.exception("session failed")


async def _aserve(args: argparse.Namespace) -> None:
    config = Config.load(args.config)
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = WebServer(config)

    threading.Thread(target=_serve_static, args=(args.host, args.http_port), daemon=True).start()
    logger.info("open http://%s:%d in a browser", args.host, args.http_port)

    async with websockets.serve(server.handle_connection, args.host, args.ws_port):
        logger.info("audio websocket listening on ws://%s:%d", args.host, args.ws_port)
        await asyncio.Future()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(_aserve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
