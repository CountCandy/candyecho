"""The optional engines (currently Drama 3) as CandyEcho sees them.

Each ``Engine`` bundles the three things the rest of the app needs to treat a
remote model like the built-in one: where its weights live (``ModelStore``),
the server that runs it (``EngineServer``) and a generator that speaks to that
server (``RemoteAudioGenerator``).

Echo is not in this registry. It is loaded in-process at startup and is always
available, so it only appears in the status listing as a fixed entry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .drama_markup import prepare_drama_chunks, strip_drama_markup
from .engine_servers import STATE_READY, EngineServer, load_engine_configs
from .model_store import STATE_READY as MODEL_READY
from .model_store import ModelStore
from .remote_engines import (
    REMOTE_ID_BASE,
    DramaClient,
    RemoteAudioGenerator,
)

logger = logging.getLogger(__name__)

ECHO_KEY = "echo"
ECHO_SAMPLE_RATE = 44_100


@dataclass
class Engine:
    key: str
    label: str
    blurb: str
    sample_rate: int
    model_key: str
    server: EngineServer
    generator: RemoteAudioGenerator
    # Names of the request fields this engine understands.
    params: tuple[str, ...]
    supports_tags: bool = False
    # (min, max) characters per chunk. Larger than Echo's because these engines
    # have no 30-second cap, and fewer chunks means fewer audible joins.
    chunk_chars: tuple[int, int] = (300, 450)
    # With best-of-N on, chunks must fit the verifier's 30-second window.
    verify_chunk_chars: tuple[int, int] = (220, 330)
    # How many further recordings of the speaker can be sent with the main one.
    max_extra_references: int = 0

    def chunk_limits(self, verify: bool) -> tuple[int, int]:
        return self.verify_chunk_chars if verify else self.chunk_chars

    def spoken_text(self, text: str) -> str:
        """Chunk text with engine markup removed (for subtitles)."""
        return strip_drama_markup(text) if self.supports_tags else text


def build_engines(
    store: ModelStore,
    log_dir: Path | str = "logs",
    overrides_path: Path | str = "engines.json",
) -> dict[str, Engine]:
    model_dirs = {"drama3": store.model_dir("drama3")}
    configs = load_engine_configs(model_dirs, overrides_path=overrides_path)

    drama_client = DramaClient(configs["drama3"].url)

    return {
        "drama3": Engine(
            key="drama3",
            label="Fish Audio Drama 3",
            blurb="Directable performances: bracketed directions, sounds and word tags.",
            sample_rate=drama_client.sample_rate,
            model_key="drama3",
            server=EngineServer(configs["drama3"], log_dir=log_dir),
            generator=RemoteAudioGenerator(
                drama_client,
                id_base=REMOTE_ID_BASE,
                prepare_chunks=prepare_drama_chunks,
                spoken_text=strip_drama_markup,
            ),
            params=DramaClient.PARAMS,
            supports_tags=True,
            # The server splits a long request itself and carries context
            # between its pieces, so a big chunk here keeps delivery continuous.
            chunk_chars=(450, 700),
            max_extra_references=8,
        ),
    }


def echo_status(loaded: bool = True) -> dict[str, Any]:
    return {
        # False while Echo is parked off the GPU to free VRAM.
        "loaded": loaded,
        "max_extra_references": 0,
        "key": ECHO_KEY,
        "label": "Echo-TTS",
        "blurb": "Built in. Clones from audio alone and keeps long reads seamless.",
        "sample_rate": ECHO_SAMPLE_RATE,
        "needs_transcript": False,
        "supports_tags": False,
        "params": ["steps", "cfg_text", "cfg_speaker", "truncation", "speaker_force"],
        "ready": True,
        "model": None,
        "server": None,
    }


def engine_status(engine: Engine, store: ModelStore, autostart: bool = False) -> dict[str, Any]:
    """Everything the UI needs to draw one engine's card (does a health probe)."""
    model = store.status(engine.model_key)
    server = engine.server.state()
    return {
        "key": engine.key,
        "label": engine.label,
        "blurb": engine.blurb,
        "sample_rate": engine.sample_rate,
        "needs_transcript": True,
        "supports_tags": engine.supports_tags,
        "max_extra_references": engine.max_extra_references,
        # Started as soon as CandyEcho launches, if its weights are present.
        "autostart": autostart,
        "params": list(engine.params),
        # An externally started server is usable even if CandyEcho did not
        # download the weights itself.
        "ready": server["state"] == STATE_READY,
        "can_generate": server["state"] == STATE_READY or model["state"] == MODEL_READY,
        "model": model,
        "server": server,
    }
