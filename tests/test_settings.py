"""Persistent preferences."""
import json

from longecho.settings import Settings


def test_defaults_when_no_file(tmp_path):
    settings = Settings(tmp_path / "s.json")
    assert settings.autostart == [] and settings.model_dirs == {}


def test_autostart_round_trip(tmp_path):
    path = tmp_path / "s.json"
    settings = Settings(path)
    assert settings.set_autostart("drama3", True) == ["drama3"]
    assert settings.set_autostart("drama3", True) == ["drama3"]     # no duplicates
    assert Settings(path).autostart == ["drama3"]                   # survives a restart
    assert settings.set_autostart("drama3", False) == []
    assert json.loads(path.read_text())["autostart"] == []


def test_other_keys_are_preserved_on_write(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"model_dirs": {"drama3": "/somewhere/drama3"}, "future": 1}))
    settings = Settings(path)
    settings.set_autostart("drama3", True)
    data = json.loads(path.read_text())
    assert data["future"] == 1 and data["model_dirs"] == {"drama3": "/somewhere/drama3"}
    assert str(settings.model_dirs["drama3"]).replace("\\", "/").endswith("somewhere/drama3")


def test_damaged_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert Settings(path).autostart == []
    path.write_text("[1, 2]")
    assert Settings(path).autostart == []
