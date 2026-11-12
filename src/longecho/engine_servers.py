"""Start, stop and watch the inference servers of the optional engines.

An optional engine such as Drama 3 cannot be loaded into CandyEcho's own
process: it pins a different PyTorch and ``transformers`` than Echo does, and is
Linux-only. It ships its own HTTP server instead, so CandyEcho launches that as
a child process (inside WSL on Windows) and talks to it over localhost.

The health endpoint is the single source of truth. A server that answers is
"ready" whether CandyEcho started it, a previous CandyEcho session left it
running, or the user launched it by hand, so none of those cases needs a
restart or special handling.

Launch commands are plain data and can be overridden per engine in an optional
``engines.json`` beside the project (see ``load_engine_configs``), which is the
escape hatch for Docker, a remote box, or a non-default install location.
Stdlib-only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

STATE_STOPPED = "stopped"
STATE_STARTING = "starting"
STATE_READY = "ready"
STATE_FAILED = "failed"

# Where scripts/setup_engines.sh installs the engines (inside WSL on Windows).
ENGINE_HOME = "~/candyecho-engines"


def to_wsl_path(path: Path | str) -> str:
    r"""Translate a Windows path to the path WSL sees it at.

    ``C:\models\drama3`` -> ``/mnt/c/models/drama3``; a ``\\wsl$\Distro\...`` or
    ``\\wsl.localhost\Distro\...`` path maps back to the Linux path inside the
    distro. Anything already POSIX is returned unchanged.
    """
    text = str(path)
    unc = re.match(r"^[\\/]{2}wsl(?:\$|\.localhost)[\\/][^\\/]+(.*)$", text, re.IGNORECASE)
    if unc:
        rest = unc.group(1).replace("\\", "/")
        return rest or "/"
    drive = re.match(r"^([A-Za-z]):[\\/]?(.*)$", text)
    if drive:
        rest = PureWindowsPath(drive.group(2)).as_posix() if drive.group(2) else ""
        rest = rest.strip("/")
        return f"/mnt/{drive.group(1).lower()}" + (f"/{rest}" if rest else "")
    return text.replace("\\", "/")


@dataclass
class EngineServerConfig:
    key: str
    label: str
    url: str
    health_path: str
    start_cmd: list[str] | None = None
    stop_cmd: list[str] | None = None
    startup_timeout: float = 1800.0
    # Shown when the launch fails, so the user knows what the one-time step is.
    setup_hint: str = ""

    @property
    def port(self) -> int:
        parsed = urlparse(self.url)
        return parsed.port or (443 if parsed.scheme == "https" else 80)


def _shell(script: str, windows: bool) -> list[str]:
    """Run a bash script natively, or inside WSL when on Windows."""
    return (["wsl.exe", "--", "bash", "-lc", script] if windows else ["bash", "-lc", script])


def default_configs(
    model_dirs: dict[str, Path],
    windows: bool | None = None,
) -> dict[str, EngineServerConfig]:
    """Built-in launch recipes, matching what scripts/setup_engines.sh installs."""
    windows = (os.name == "nt") if windows is None else windows

    def linux_path(p: Path) -> str:
        return to_wsl_path(p.resolve()) if windows else str(p.resolve())

    hint = (
        "One-time setup: open a WSL terminal in the CandyEcho folder and run "
        "`bash scripts/setup_engines.sh`."
        if windows
        else "One-time setup: run `bash scripts/setup_engines.sh`."
    )

    drama_url = os.environ.get("CANDYECHO_DRAMA3_URL", "http://127.0.0.1:8080")
    drama_port = urlparse(drama_url).port or 8080

    # Natively the server listens on loopback only. Under WSL it binds all
    # interfaces of the WSL VM, which is what Windows' localhost forwarding
    # reaches most reliably; with WSL's default NAT networking that is not
    # exposed to the LAN.
    bind = "0.0.0.0" if windows else "127.0.0.1"

    drama_dir = linux_path(model_dirs["drama3"])
    drama_script = (
        f"cd {ENGINE_HOME}/fish-speech && exec .venv/bin/python tools/api_server.py "
        f"--listen {bind}:{drama_port} "
        f"--llama-checkpoint-path {shlex.quote(drama_dir + '/drama-3')} "
        f"--decoder-checkpoint-path {shlex.quote(drama_dir + '/s2-pro/codec.pth')} "
        "--device cuda:0"
    )
    def stop(pattern: str) -> list[str] | None:
        # Killing wsl.exe does not reliably end the Linux process behind it, so
        # on Windows the server is stopped from inside WSL by name. Natively the
        # child is the server itself (bash is replaced by `exec`).
        return ["wsl.exe", "--", "pkill", "-f", pattern] if windows else None

    return {
        "drama3": EngineServerConfig(
            key="drama3",
            label="Fish Audio Drama 3",
            url=drama_url,
            health_path="/v1/health",
            start_cmd=_shell(drama_script, windows),
            stop_cmd=stop("tools/api_server.py"),
            setup_hint=hint,
        ),
    }


def load_engine_configs(
    model_dirs: dict[str, Path],
    overrides_path: Path | str = "engines.json",
    windows: bool | None = None,
) -> dict[str, EngineServerConfig]:
    """Defaults, with any per-engine overrides from ``engines.json`` applied.

    The file maps an engine key to any of ``url``, ``health_path``,
    ``start_cmd``, ``stop_cmd`` and ``startup_timeout``. Commands are argument
    lists; ``{model_dir}`` and ``{model_dir_wsl}`` expand to that engine's
    weights folder. Set ``"start_cmd": null`` for a server you run yourself.
    """
    configs = default_configs(model_dirs, windows=windows)
    path = Path(overrides_path)
    if not path.is_file():
        return configs
    try:
        overrides = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.error(f"Ignoring {path}: {e}")
        return configs
    if not isinstance(overrides, dict):
        logger.error(f"Ignoring {path}: expected an object keyed by engine name")
        return configs

    for key, entry in overrides.items():
        cfg = configs.get(key)
        if cfg is None or not isinstance(entry, dict):
            logger.warning(f"{path}: unknown engine '{key}', skipped")
            continue
        model_dir = model_dirs[key].resolve()
        subs = {"model_dir": str(model_dir), "model_dir_wsl": to_wsl_path(model_dir)}

        def expand(cmd: Any) -> list[str] | None:
            if cmd is None:
                return None
            if not isinstance(cmd, list) or not all(isinstance(a, str) for a in cmd):
                raise ValueError("commands must be lists of strings")
            out = []
            for arg in cmd:
                for name, value in subs.items():
                    arg = arg.replace("{" + name + "}", value)
                out.append(arg)
            return out

        try:
            if "url" in entry:
                cfg.url = str(entry["url"]).rstrip("/")
            if "health_path" in entry:
                cfg.health_path = str(entry["health_path"])
            if "start_cmd" in entry:
                cfg.start_cmd = expand(entry["start_cmd"])
            if "stop_cmd" in entry:
                cfg.stop_cmd = expand(entry["stop_cmd"])
            if "startup_timeout" in entry:
                cfg.startup_timeout = float(entry["startup_timeout"])
        except (TypeError, ValueError) as e:
            logger.error(f"{path}: bad settings for '{key}' ignored ({e})")
    return configs


def http_probe(url: str, timeout: float = 1.5) -> bool:
    """True if the URL answers with a 2xx."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


@dataclass
class _Run:
    process: Any = None
    started: float = 0.0
    error: str | None = None
    starting: bool = False
    log_path: Path | None = None
    log_file: Any = field(default=None, repr=False)
    watcher: threading.Thread | None = field(default=None, repr=False)
    # Set by stop(): the watcher must exit without reporting a failure.
    cancelled: bool = False


class EngineServer:
    """One engine's inference server: launch it, watch it, stop it."""

    def __init__(
        self,
        config: EngineServerConfig,
        log_dir: Path | str = "logs",
        popen: Callable[..., Any] = subprocess.Popen,
        probe: Callable[[str], bool] = http_probe,
        poll_interval: float = 2.0,
    ):
        self.config = config
        self.log_dir = Path(log_dir)
        self._popen = popen
        self._probe = probe
        self._poll_interval = poll_interval
        self._run = _Run()
        self._lock = threading.Lock()
        # How long the last launch took to become ready (None until one has).
        self.last_load_seconds: int | None = None

    @property
    def health_url(self) -> str:
        return self.config.url.rstrip("/") + self.config.health_path

    def is_healthy(self) -> bool:
        return self._probe(self.health_url)

    def is_running(self) -> bool:
        """True while a server process this object launched is alive."""
        with self._lock:
            return self._process_alive()

    def _process_alive(self) -> bool:
        proc = self._run.process
        return proc is not None and proc.poll() is None

    def log_tail(self, lines: int = 12) -> str:
        path = self._run.log_path
        if path is None or not path.is_file():
            return ""
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def state(self) -> dict[str, Any]:
        healthy = self.is_healthy()
        with self._lock:
            run = self._run
            alive = self._process_alive()
            if healthy:
                run.error = None
                run.starting = False
                state = STATE_READY
            elif run.starting and alive:
                state = STATE_STARTING
            elif run.error:
                state = STATE_FAILED
            else:
                state = STATE_STOPPED
            return {
                "state": state,
                "url": self.config.url,
                "managed": alive,
                "can_start": self.config.start_cmd is not None,
                "error": run.error if state == STATE_FAILED else None,
                "seconds_starting": round(time.time() - run.started) if state == STATE_STARTING else None,
                "last_load_seconds": self.last_load_seconds,
            }

    def start(self) -> str:
        """Launch the server if it is not already up. Returns the resulting state."""
        if self.is_healthy():
            return STATE_READY
        with self._lock:
            if self._run.starting and self._process_alive():
                return STATE_STARTING
            if self.config.start_cmd is None:
                self._run.error = (
                    f"No launch command is configured for {self.config.label}. Start its "
                    f"server yourself so that it answers at {self.config.url}."
                )
                return STATE_FAILED

            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"engine-{self.config.key}.log"
            run = _Run(started=time.time(), starting=True, log_path=log_path)
            try:
                run.log_file = open(log_path, "wb")
                run.process = self._popen(
                    self.config.start_cmd,
                    stdout=run.log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                launcher = self.config.start_cmd[0]
                extra = (
                    " WSL does not appear to be installed: run `wsl --install` in an "
                    "administrator terminal and reboot."
                    if launcher.lower().startswith("wsl")
                    else ""
                )
                run.starting = False
                run.error = f"Could not run '{launcher}'.{extra} {self.config.setup_hint}".strip()
                self._close_log(run)
                self._run = run
                return STATE_FAILED
            except OSError as e:
                run.starting = False
                run.error = f"Could not launch {self.config.label}: {e}"
                self._close_log(run)
                self._run = run
                return STATE_FAILED
            self._run = run

        logger.info(f"Starting {self.config.label} server: {' '.join(self.config.start_cmd)}")
        run.watcher = threading.Thread(
            target=self._watch, args=(run, run.process), name=f"engine-{self.config.key}", daemon=True
        )
        run.watcher.start()
        return STATE_STARTING

    @staticmethod
    def _close_log(run: _Run) -> None:
        if run.log_file is not None:
            try:
                run.log_file.close()
            except OSError:
                pass
            run.log_file = None

    def _watch(self, run: _Run, process: Any) -> None:
        """Wait for the launched server to answer, or to die trying."""
        deadline = run.started + self.config.startup_timeout
        while True:
            if run.cancelled:
                return
            if self._probe(self.health_url):
                with self._lock:
                    run.starting = False
                    run.error = None
                self.last_load_seconds = round(time.time() - run.started)
                logger.info(
                    f"{self.config.label} server is ready "
                    f"({self.last_load_seconds}s to load)"
                )
                return
            code = process.poll()
            if code is not None:
                tail = self.log_tail()
                with self._lock:
                    if self._run is run and not run.cancelled:
                        run.starting = False
                        run.error = (
                            f"{self.config.label} server exited (code {code}) before it "
                            f"was ready. {self.config.setup_hint}"
                            + (f"\n\nLast output:\n{tail}" if tail else "")
                        ).strip()
                    self._close_log(run)
                logger.error(f"{self.config.label} server exited with code {code}")
                return
            if time.time() > deadline:
                with self._lock:
                    run.starting = False
                    run.error = (
                        f"{self.config.label} server did not answer within "
                        f"{self.config.startup_timeout / 60:.0f} minutes."
                    )
                self._terminate(run)
                return
            time.sleep(self._poll_interval)

    def wait_ready(self, timeout: float | None = None) -> None:
        """Block until the server answers. Raises RuntimeError if it cannot start."""
        limit = time.time() + (timeout if timeout is not None else self.config.startup_timeout)
        state = self.start()
        while True:
            if state == STATE_READY or self.is_healthy():
                return
            with self._lock:
                error = self._run.error
                still_starting = self._run.starting and self._process_alive()
                watcher = self._run.watcher
            if not error and not still_starting and watcher is not None:
                # The process has gone; let the watcher finish writing up why.
                watcher.join(timeout=10)
                with self._lock:
                    error = self._run.error
            if error:
                raise RuntimeError(error)
            if not still_starting and not self.is_healthy():
                raise RuntimeError(f"{self.config.label} server stopped unexpectedly.")
            if time.time() > limit:
                raise RuntimeError(f"Timed out waiting for the {self.config.label} server.")
            time.sleep(self._poll_interval)
            state = STATE_STARTING

    def _terminate(self, run: _Run) -> None:
        proc = run.process
        if self.config.stop_cmd:
            try:
                subprocess.run(
                    self.config.stop_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as e:
                logger.warning(f"Stop command for {self.config.label} failed: {e}")
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except OSError:
                pass
        self._close_log(run)

    def stop(self) -> None:
        """Stop a server this process launched. Externally started ones are left alone."""
        with self._lock:
            run = self._run
            run.starting = False
            run.error = None
            run.cancelled = True
            owned = run.process is not None
        if owned:
            logger.info(f"Stopping {self.config.label} server")
            self._terminate(run)
            with self._lock:
                run.process = None
