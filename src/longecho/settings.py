"""Small persistent preferences that must be known before the page opens.

Most preferences live in the browser. These cannot: which engines to start as
soon as CandyEcho launches, and where a model's weights live if not under the
default ``models/`` folder. They are kept in ``candyecho-settings.json`` beside
the project (git-ignored), written atomically, and tolerate a missing or
damaged file by falling back to defaults.

Stdlib-only.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PATH = "candyecho-settings.json"


class Settings:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._read()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            logger.error(f"Ignoring {self.path}: {e}")
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- engines started at launch -------------------------------------------

    @property
    def autostart(self) -> list[str]:
        value = self._data.get("autostart")
        return [str(v) for v in value] if isinstance(value, list) else []

    def set_autostart(self, engine_key: str, enabled: bool) -> list[str]:
        with self._lock:
            current = [k for k in self.autostart if k != engine_key]
            if enabled:
                current.append(engine_key)
            self._data["autostart"] = current
            self._write()
            return current

    # -- where a model's weights live ----------------------------------------

    @property
    def model_dirs(self) -> dict[str, Path]:
        """Per-model overrides of the weights folder, e.g. a path on WSL's disk."""
        value = self._data.get("model_dirs")
        if not isinstance(value, dict):
            return {}
        return {str(k): Path(str(v)) for k, v in value.items() if v}
