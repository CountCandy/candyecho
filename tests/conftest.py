import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def tmp_voice_dir(tmp_path):
    """Create temporary voice directory"""
    voice_dir = tmp_path / "voice_library"
    voice_dir.mkdir()
    return voice_dir


@pytest.fixture(autouse=True)
def no_speaker_model_download():
    """Generations rate their output against the sample with a speaker model.
    Tests must not fetch it; one that wants scores patches this itself."""
    try:
        import longecho.main as main
    except Exception:           # stdlib-only test runs without the app's deps
        yield
        return
    main.app.state.voice_scoring_failed = False
    with patch.object(main, "get_speaker_similarity", side_effect=RuntimeError("off in tests")):
        yield
    main.app.state.voice_scoring_failed = False
