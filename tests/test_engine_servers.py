"""Engine server lifecycle: launching, health, failure reporting, config."""
import json
import time

import pytest

from longecho.engine_servers import (
    EngineServer,
    EngineServerConfig,
    default_configs,
    load_engine_configs,
    to_wsl_path,
)


class FakeProcess:
    def __init__(self, exit_code=None):
        self.exit_code = exit_code
        self.terminated = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        self.exit_code = -15

    def wait(self, timeout=None):
        return self.exit_code

    def kill(self):
        self.exit_code = -9


def make_server(tmp_path, healthy, process=None, start_cmd=("run-engine",), popen=None):
    cfg = EngineServerConfig(
        key="toy", label="Toy", url="http://127.0.0.1:9", health_path="/health",
        start_cmd=list(start_cmd) if start_cmd else None, setup_hint="Run the setup script.",
        startup_timeout=5,
    )
    launched = []

    def default_popen(cmd, **kwargs):
        launched.append(cmd)
        kwargs["stdout"].write(b"line one\nCUDA out of memory\n")
        kwargs["stdout"].flush()
        return process or FakeProcess()

    server = EngineServer(
        cfg, log_dir=tmp_path / "logs", popen=popen or default_popen,
        probe=lambda url: healthy(), poll_interval=0.01,
    )
    return server, launched


def wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.parametrize(
    "windows, expected",
    [
        (r"C:\Users\Me\candyecho\models\drama3", "/mnt/c/Users/Me/candyecho/models/drama3"),
        ("D:/AI Models/breeze", "/mnt/d/AI Models/breeze"),
        ("C:\\", "/mnt/c"),
        (r"\\wsl$\Ubuntu\home\me\models", "/home/me/models"),
        (r"\\wsl.localhost\Ubuntu-24.04\home\me", "/home/me"),
        ("/already/posix", "/already/posix"),
    ],
)
def test_to_wsl_path(windows, expected):
    assert to_wsl_path(windows) == expected


def test_windows_defaults_run_inside_wsl_with_translated_paths(tmp_path):
    # Path translation itself is covered by test_to_wsl_path; this checks the
    # shape of the command (a real Windows path cannot be resolved on Linux CI).
    dirs = {"drama3": tmp_path / "drama3", "breeze": tmp_path / "breeze"}
    cfgs = default_configs(dirs, windows=True)
    drama = cfgs["drama3"]
    assert drama.start_cmd[:4] == ["wsl.exe", "--", "bash", "-lc"]
    script = drama.start_cmd[4]
    assert "tools/api_server.py" in script and "--listen 0.0.0.0:8080" in script
    assert "/drama-3" in script and "/s2-pro/codec.pth" in script and "--device cuda:0" in script
    assert drama.stop_cmd == ["wsl.exe", "--", "pkill", "-f", "tools/api_server.py"]
    assert drama.health_path == "/v1/health"

    breeze = cfgs["breeze"]
    assert "breeze_infer.api" in breeze.start_cmd[4] and "--port 7860" in breeze.start_cmd[4]
    assert breeze.health_path == "/health"


def test_native_defaults_use_bash_directly(tmp_path):
    cfgs = default_configs({"drama3": tmp_path / "d", "breeze": tmp_path / "b"}, windows=False)
    assert cfgs["drama3"].start_cmd[:2] == ["bash", "-lc"]
    assert cfgs["drama3"].stop_cmd is None
    # Loopback only when not going through WSL.
    assert "--listen 127.0.0.1:8080" in cfgs["drama3"].start_cmd[2]
    assert "--host 127.0.0.1" in cfgs["breeze"].start_cmd[2]
    assert str((tmp_path / "d").resolve()) in cfgs["drama3"].start_cmd[2]


def test_paths_with_spaces_are_quoted(tmp_path):
    spaced = tmp_path / "my models" / "drama3"
    cfgs = default_configs({"drama3": spaced, "breeze": tmp_path / "b"}, windows=False)
    assert f"'{spaced.resolve()}/drama-3'" in cfgs["drama3"].start_cmd[2]


def test_overrides_file_replaces_commands_and_urls(tmp_path):
    dirs = {"drama3": tmp_path / "d", "breeze": tmp_path / "b"}
    overrides = tmp_path / "engines.json"
    overrides.write_text(json.dumps({
        "drama3": {
            "url": "http://10.0.0.5:9000/",
            "start_cmd": ["docker", "run", "-v", "{model_dir}:/app/checkpoints", "img"],
            "stop_cmd": ["docker", "stop", "drama"],
            "startup_timeout": 60,
        },
        "breeze": {"start_cmd": None},
        "mystery": {"url": "x"},
    }))
    cfgs = load_engine_configs(dirs, overrides_path=overrides, windows=False)
    assert cfgs["drama3"].url == "http://10.0.0.5:9000"
    assert cfgs["drama3"].port == 9000
    assert cfgs["drama3"].start_cmd[3] == f"{(tmp_path / 'd').resolve()}:/app/checkpoints"
    assert cfgs["drama3"].startup_timeout == 60
    assert cfgs["breeze"].start_cmd is None


def test_bad_overrides_file_falls_back_to_defaults(tmp_path):
    dirs = {"drama3": tmp_path / "d", "breeze": tmp_path / "b"}
    overrides = tmp_path / "engines.json"
    overrides.write_text("{broken")
    cfgs = load_engine_configs(dirs, overrides_path=overrides, windows=False)
    assert cfgs["drama3"].start_cmd[0] == "bash"


def test_already_running_server_is_ready_without_launching(tmp_path):
    server, launched = make_server(tmp_path, healthy=lambda: True)
    assert server.start() == "ready"
    assert launched == []
    state = server.state()
    assert state["state"] == "ready" and state["managed"] is False


def test_start_then_becomes_ready(tmp_path):
    up = {"ok": False}
    server, launched = make_server(tmp_path, healthy=lambda: up["ok"])
    assert server.state()["state"] == "stopped"
    assert server.start() == "starting"
    assert launched == [["run-engine"]]
    assert server.state()["state"] == "starting"
    # A second click while it is loading must not launch a second copy.
    assert server.start() == "starting"
    assert len(launched) == 1
    up["ok"] = True
    state = server.state()
    assert state["state"] == "ready" and state["managed"] is True


def test_crash_during_startup_reports_the_log_tail(tmp_path):
    server, _ = make_server(tmp_path, healthy=lambda: False, process=FakeProcess(exit_code=1))
    server.start()
    assert wait_for(lambda: server.state()["state"] == "failed")
    error = server.state()["error"]
    assert "exited (code 1)" in error
    assert "CUDA out of memory" in error
    assert "Run the setup script." in error


def test_missing_launcher_explains_itself(tmp_path):
    def no_such_program(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    server, _ = make_server(
        tmp_path, healthy=lambda: False, start_cmd=("wsl.exe", "--", "bash"), popen=no_such_program
    )
    assert server.start() == "failed"
    error = server.state()["error"]
    assert "wsl --install" in error


def test_no_launch_command_asks_the_user_to_start_it(tmp_path):
    server, _ = make_server(tmp_path, healthy=lambda: False, start_cmd=None)
    assert server.start() == "failed"
    assert "http://127.0.0.1:9" in server.state()["error"]
    assert server.state()["can_start"] is False


def test_wait_ready_returns_once_healthy(tmp_path):
    calls = {"n": 0}

    def healthy():
        calls["n"] += 1
        return calls["n"] > 4

    server, launched = make_server(tmp_path, healthy=healthy)
    server.wait_ready(timeout=5)
    assert len(launched) == 1


def test_wait_ready_raises_with_the_failure(tmp_path):
    server, _ = make_server(tmp_path, healthy=lambda: False, process=FakeProcess(exit_code=2))
    with pytest.raises(RuntimeError, match="exited"):
        server.wait_ready(timeout=5)


def test_stop_terminates_only_what_it_started(tmp_path):
    process = FakeProcess()
    server, _ = make_server(tmp_path, healthy=lambda: False, process=process)
    server.start()
    server.stop()
    assert process.terminated
    assert server.state()["state"] == "stopped"

    external, _ = make_server(tmp_path, healthy=lambda: True)
    external.stop()  # nothing launched: must be a no-op
    assert external.state()["state"] == "ready"
