from __future__ import annotations

import argparse
import asyncio
import logging

from .config import Config
from .orchestrator import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="chatterbox local speech agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


async def _amain(config: Config) -> None:
    agent = Agent(config)
    await agent.run()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_amain(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
