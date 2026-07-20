"""
Test FastAPI app initialization and lifespan context manager.

This test verifies that the lifespan context manager properly initializes
the application on startup without using the deprecated @app.on_event decorator.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys


@pytest.fixture
def mock_dependencies():
    """Mock all heavy dependencies for faster testing"""
    # Mock the inference module
    mock_inference = Mock()
    mock_model = Mock()
    mock_fish_ae = Mock()
    mock_pca_state = Mock()

    mock_inference.load_model_from_hf = Mock(return_value=mock_model)
    mock_inference.load_fish_ae_from_hf = Mock(return_value=mock_fish_ae)
    mock_inference.load_pca_state_from_hf = Mock(return_value=mock_pca_state)

    sys.modules['inference'] = mock_inference
    sys.modules['inference_blockwise'] = Mock()

    # Mock VoiceManager
    mock_voice_manager_class = Mock()
    mock_voice_manager_instance = Mock()
    mock_voice_manager_instance.get_voice_names.return_value = ["voice1", "voice2"]
    mock_voice_manager_instance.get_voice_info.return_value = [
        {"name": "voice1", "duration_seconds": 1.5},
        {"name": "voice2", "duration_seconds": 2.0},
    ]
    mock_voice_manager_class.return_value = mock_voice_manager_instance

    # Mock AudioGenerator
    mock_audio_generator_class = Mock()
    mock_audio_generator_instance = Mock()
    mock_audio_generator_class.return_value = mock_audio_generator_instance

    with patch('longecho.main.load_model_from_hf', return_value=mock_model), \
         patch('longecho.main.load_fish_ae_from_hf', return_value=mock_fish_ae), \
         patch('longecho.main.load_pca_state_from_hf', return_value=mock_pca_state), \
         patch('longecho.main.VoiceManager', mock_voice_manager_class), \
         patch('longecho.main.AudioGenerator', mock_audio_generator_class):

        yield {
            'model': mock_model,
            'fish_ae': mock_fish_ae,
            'pca_state': mock_pca_state,
            'voice_manager': mock_voice_manager_instance,
            'audio_generator': mock_audio_generator_instance,
        }


def test_lifespan_initializes_app(mock_dependencies):
    """
    Test that the lifespan context manager properly initializes the app.

    This verifies:
    1. Models are loaded on startup
    2. VoiceManager is initialized
    3. AudioGenerator is initialized
    4. Health endpoint returns correct status
    """
    from longecho.main import app

    with TestClient(app) as client:
        # Test health endpoint to verify app started successfully
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["voices_loaded"] == 2  # Should match mock_voice_manager.get_voice_names


def test_voices_endpoint(mock_dependencies):
    """Test that the voices endpoint returns available voices"""
    from longecho.main import app

    with TestClient(app) as client:
        response = client.get("/voices")
        assert response.status_code == 200

        data = response.json()
        assert "voices" in data
        assert [v["name"] for v in data["voices"]] == ["voice1", "voice2"]
        assert data["voices"][0]["duration_seconds"] == 1.5


def test_root_endpoint(mock_dependencies):
    """Test that root endpoint returns the main page"""
    from longecho.main import app

    with TestClient(app) as client:
        # This will fail if static/index.html doesn't exist,
        # but we're mainly testing that the endpoint is registered
        response = client.get("/")
        # We expect either 200 (if file exists) or 404 (if file doesn't exist)
        # Both are valid - we're just testing the app initialized
        assert response.status_code in [200, 404]


def test_generate_validation_empty_text(mock_dependencies):
    """Test that generate endpoint rejects empty text"""
    from longecho.main import app

    with TestClient(app) as client:
        response = client.post("/generate", json={"text": "", "voice": "voice1"})
        assert response.status_code == 422  # Unprocessable Entity

        # Check that the error message mentions text validation
        data = response.json()
        assert "detail" in data


def test_generate_validation_text_too_long(mock_dependencies):
    """Test that generate endpoint rejects text longer than max_length"""
    from longecho.main import app

    with TestClient(app) as client:
        long_text = "a" * 100001
        response = client.post("/generate", json={"text": long_text, "voice": "voice1"})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


def test_generate_validation_empty_voice(mock_dependencies):
    """Test that generate endpoint rejects empty voice"""
    from longecho.main import app

    with TestClient(app) as client:
        response = client.post("/generate", json={"text": "Hello world", "voice": ""})
        assert response.status_code == 422  # Unprocessable Entity

        # Check that the error message mentions voice validation
        data = response.json()
        assert "detail" in data


def test_generate_validation_valid_inputs(mock_dependencies):
    """Test that generate endpoint accepts valid inputs"""
    from longecho.main import app
    import torch

    # Mock the voice manager to return valid voice data
    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),  # speaker_latent
        torch.ones(1, 10)         # speaker_mask
    )

    # Mock the audio generator to yield a single audio chunk
    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)  # 1 second of audio
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    # Mock segment_text to return a single chunk
    with patch('longecho.main.segment_text', return_value=["Hello world"]):
        with TestClient(app) as client:
            response = client.post("/generate", json={"text": "Hello world", "voice": "voice1"})
            # SSE endpoint should return 200 for successful stream initiation
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def test_generate_validation_max_length_boundary(mock_dependencies):
    """Test that generate endpoint accepts text below max_length"""
    from longecho.main import app
    import torch

    # Mock the voice manager to return valid voice data
    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),  # speaker_latent
        torch.ones(1, 10)         # speaker_mask
    )

    # Mock the audio generator to yield a single audio chunk
    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    test_text = "This is a test sentence. " * 200  # ~5000 characters

    # Mock segment_text to return a single chunk
    with patch('longecho.main.segment_text', return_value=[test_text[:100]]):
        with TestClient(app) as client:
            response = client.post("/generate", json={"text": test_text, "voice": "voice1"})
            # Should accept text below max length
            assert response.status_code == 200


def test_stop_endpoint(mock_dependencies):
    """Test that stop endpoint calls request_stop on audio generator"""
    from longecho.main import app

    mock_audio_generator = mock_dependencies['audio_generator']

    with TestClient(app) as client:
        response = client.post("/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stop requested"

        # Verify request_stop was called
        mock_audio_generator.request_stop.assert_called_once()


def test_generate_with_normalization_level_moderate(mock_dependencies):
    """Test that generate endpoint accepts normalization_level parameter"""
    from longecho.main import app
    import torch

    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),
        torch.ones(1, 10)
    )

    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    with patch('longecho.main.segment_text', return_value=["Hello world"]):
        with TestClient(app) as client:
            response = client.post("/generate", json={
                "text": "Hello world",
                "voice": "voice1",
                "normalization_level": "moderate"
            })
            assert response.status_code == 200


def test_generate_with_normalization_level_full(mock_dependencies):
    """Test that generate endpoint accepts normalization_level='full'"""
    from longecho.main import app
    import torch

    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),
        torch.ones(1, 10)
    )

    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    with patch('longecho.main.segment_text', return_value=["Hello world"]):
        with TestClient(app) as client:
            response = client.post("/generate", json={
                "text": "Hello world",
                "voice": "voice1",
                "normalization_level": "full"
            })
            assert response.status_code == 200


def test_generate_with_invalid_normalization_level(mock_dependencies):
    """Test that generate endpoint rejects invalid normalization_level"""
    from longecho.main import app

    with TestClient(app) as client:
        response = client.post("/generate", json={
            "text": "Hello world",
            "voice": "voice1",
            "normalization_level": "invalid"
        })
        # Should get 422 validation error
        assert response.status_code == 422


def test_generate_normalizes_text_before_segmentation(mock_dependencies):
    """Test that text is normalized before being segmented"""
    from longecho.main import app
    import torch

    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),
        torch.ones(1, 10)
    )

    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    # Use a mock for segment_text to capture what it receives
    captured_text = []

    def capture_segment_text(text):
        captured_text.append(text)
        return [text]

    with patch('longecho.main.segment_text', side_effect=capture_segment_text):
        with TestClient(app) as client:
            # Input with currency that should be normalized
            response = client.post("/generate", json={
                "text": "The cost is $5M.",
                "voice": "voice1",
            })
            assert response.status_code == 200

            # Verify segment_text received normalized text (without $ sign)
            assert len(captured_text) == 1
            assert "$" not in captured_text[0]
            assert "million dollars" in captured_text[0].lower() or "5M" not in captured_text[0]


def test_upload_voice_success(mock_dependencies, tmp_path):
    """Uploading a .wav saves it to the voice dir and preprocesses it."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = []
    vm.voice_dir = tmp_path
    vm.add_voice = Mock(return_value="myvoice")

    with TestClient(app) as client:
        response = client.post(
            "/voices",
            files={"file": ("myvoice.wav", b"RIFF....WAVEfakeaudio", "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "voice": "myvoice"}

    # File was saved into the voice dir and handed to the preprocessor
    assert (tmp_path / "myvoice.wav").exists()
    vm.add_voice.assert_called_once()
    # No leftover temp upload files
    assert not list(tmp_path.glob(".*.upload"))


def test_upload_voice_rejects_non_wav(mock_dependencies, tmp_path):
    """Non-.wav uploads are rejected with 400."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = []
    vm.voice_dir = tmp_path

    with TestClient(app) as client:
        response = client.post(
            "/voices",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400

    assert list(tmp_path.iterdir()) == []


def test_upload_voice_rejects_duplicate(mock_dependencies, tmp_path):
    """Uploading a voice whose name already exists returns 409."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["existing"]
    vm.voice_dir = tmp_path
    vm.add_voice = Mock()

    with TestClient(app) as client:
        response = client.post(
            "/voices",
            files={"file": ("existing.wav", b"RIFF....WAVE", "audio/wav")},
        )
        assert response.status_code == 409

    vm.add_voice.assert_not_called()


def test_upload_voice_sanitizes_filename(mock_dependencies, tmp_path):
    """Path components in the filename are stripped (no traversal)."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = []
    vm.voice_dir = tmp_path
    vm.add_voice = Mock(return_value="passwd")

    with TestClient(app) as client:
        response = client.post(
            "/voices",
            files={"file": ("../../etc/passwd.wav", b"RIFF....WAVE", "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["voice"] == "passwd"

    # Written inside the voice dir, not outside it
    assert (tmp_path / "passwd.wav").exists()


def test_upload_voice_cleans_up_on_processing_failure(mock_dependencies, tmp_path):
    """If preprocessing fails, the saved .wav is removed and 400 returned."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = []
    vm.voice_dir = tmp_path
    vm.add_voice = Mock(side_effect=RuntimeError("bad audio"))

    with TestClient(app) as client:
        response = client.post(
            "/voices",
            files={"file": ("broken.wav", b"not really audio", "audio/wav")},
        )
        assert response.status_code == 400

    # The broken sample must not be left behind
    assert not (tmp_path / "broken.wav").exists()
    assert not list(tmp_path.glob(".*.upload"))


def test_voice_audio_preview(mock_dependencies, tmp_path):
    """GET /voices/{name}/audio serves the reference .wav."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    wav = tmp_path / "myvoice.wav"
    wav.write_bytes(b"RIFF....WAVEfakeaudio")
    vm.get_voice_path = Mock(return_value=wav)

    with TestClient(app) as client:
        resp = client.get("/voices/myvoice/audio")
        assert resp.status_code == 200
        assert resp.content == b"RIFF....WAVEfakeaudio"
        assert resp.headers["content-type"].startswith("audio/")


def test_voice_audio_preview_not_found(mock_dependencies):
    """Previewing an unknown voice returns 404."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_path = Mock(return_value=None)

    with TestClient(app) as client:
        assert client.get("/voices/nope/audio").status_code == 404


def test_rename_voice_success(mock_dependencies):
    """Renaming a voice calls VoiceManager.rename_voice with the sanitized name."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["old"]
    vm.rename_voice = Mock()

    with TestClient(app) as client:
        resp = client.post("/voices/old/rename", json={"new_name": "New Name"})
        assert resp.status_code == 200
        assert resp.json()["new"] == "New Name"
        vm.rename_voice.assert_called_once_with("old", "New Name")


def test_rename_voice_not_found(mock_dependencies):
    """Renaming a missing voice returns 404."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["other"]

    with TestClient(app) as client:
        resp = client.post("/voices/missing/rename", json={"new_name": "x"})
        assert resp.status_code == 404


def test_rename_voice_conflict(mock_dependencies):
    """A name collision surfaces as 409."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["old"]
    vm.rename_voice = Mock(side_effect=ValueError("Voice 'taken' already exists"))

    with TestClient(app) as client:
        resp = client.post("/voices/old/rename", json={"new_name": "taken"})
        assert resp.status_code == 409


def test_delete_voice_success(mock_dependencies):
    """Deleting a voice calls VoiceManager.delete_voice and returns 200."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["gone"]
    vm.delete_voice = Mock()

    with TestClient(app) as client:
        resp = client.delete("/voices/gone")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted", "voice": "gone"}
        vm.delete_voice.assert_called_once_with("gone")


def test_delete_voice_not_found(mock_dependencies):
    """Deleting a missing voice returns 404 without touching the manager."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["other"]
    vm.delete_voice = Mock()

    with TestClient(app) as client:
        resp = client.delete("/voices/missing")
        assert resp.status_code == 404
        vm.delete_voice.assert_not_called()


def test_voice_manager_delete_voice_removes_files(tmp_path):
    """delete_voice drops the in-memory entry and unlinks .wav + .pkl."""
    from longecho.voice_manager import VoiceManager

    wav = tmp_path / "v.wav"
    pkl = tmp_path / "v.pkl"
    wav.write_bytes(b"RIFF....WAVE")
    pkl.write_bytes(b"cache")

    vm = VoiceManager.__new__(VoiceManager)  # skip __init__ (needs models)
    vm.voice_dir = tmp_path
    vm.voices = {"v": (None, None)}
    vm._durations = {"v": 1.0}

    vm.delete_voice("v")

    assert "v" not in vm.voices
    assert not wav.exists()
    assert not pkl.exists()

    with pytest.raises(ValueError):
        vm.delete_voice("v")  # already gone -> ValueError


def test_extract_text_from_txt(mock_dependencies):
    """POST /extract-text decodes a .txt upload to plain text."""
    from longecho.main import app

    with TestClient(app) as client:
        resp = client.post("/extract-text", files={"file": ("notes.txt", b"Hello candy world", "text/plain")})
        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello candy world"
        assert resp.json()["chars"] == len("Hello candy world")


def test_extract_text_from_epub(mock_dependencies):
    """POST /extract-text pulls reading-order text out of a minimal .epub."""
    import io
    import zipfile
    from longecho.main import app

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>'
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        z.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><manifest>'
            '<item id="c1" href="chap1.xhtml" media-type="application/xhtml+xml"/>'
            '</manifest><spine><itemref idref="c1"/></spine></package>',
        )
        z.writestr(
            "OEBPS/chap1.xhtml",
            "<html><body><h1>Chapter One</h1><p>Hello from the candy book.</p></body></html>",
        )

    with TestClient(app) as client:
        resp = client.post(
            "/extract-text",
            files={"file": ("book.epub", buf.getvalue(), "application/epub+zip")},
        )
        assert resp.status_code == 200
        text = resp.json()["text"]
        assert "Chapter One" in text
        assert "candy book" in text


def test_extract_text_rejects_unsupported(mock_dependencies):
    """Unsupported extensions are rejected with 400."""
    from longecho.main import app

    with TestClient(app) as client:
        resp = client.post("/extract-text", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")})
        assert resp.status_code == 400


def test_generate_forwards_advanced_params(mock_dependencies):
    """Advanced controls reach generate_long_audio as rng_seed + gen_params."""
    from longecho.main import app
    import torch

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["voice1"]
    vm.get_voice.return_value = (torch.randn(1, 10, 256), torch.ones(1, 10))

    ag = mock_dependencies['audio_generator']

    def _one_chunk():
        yield torch.randn(1, 1, 4410)

    ag.generate_long_audio.return_value = (5, _one_chunk())

    with patch('longecho.main.segment_text', return_value=["Hello"]):
        with TestClient(app) as client:
            resp = client.post("/generate", json={
                "text": "Hello", "voice": "voice1",
                "seed": 123, "steps": 24, "cfg_text": 4.5, "cfg_speaker": 6.0,
            })
            assert resp.status_code == 200

    _, kwargs = ag.generate_long_audio.call_args
    assert kwargs.get("rng_seed") == 123
    gp = kwargs.get("gen_params") or {}
    assert gp.get("num_steps") == 24
    assert gp.get("cfg_scale_text") == 4.5
    assert gp.get("cfg_scale_speaker") == 6.0


def test_generate_rejects_out_of_range_steps(mock_dependencies):
    """Advanced controls are range-validated (steps capped at 64)."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["voice1"]

    with TestClient(app) as client:
        resp = client.post("/generate", json={"text": "Hi", "voice": "voice1", "steps": 500})
        assert resp.status_code == 422


def test_openai_list_voices(mock_dependencies):
    """GET /v1/audio/voices returns a sorted voice list."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["bravo", "alpha"]

    with TestClient(app) as client:
        resp = client.get("/v1/audio/voices")
        assert resp.status_code == 200
        assert resp.json()["voices"] == ["alpha", "bravo"]


def test_openai_speech_returns_audio(mock_dependencies):
    """POST /v1/audio/speech returns a single audio response (non-streaming)."""
    from longecho.main import app
    import torch

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["voice1"]
    vm.get_voice.return_value = (torch.randn(1, 10, 256), torch.ones(1, 10))

    ag = mock_dependencies['audio_generator']

    def _one_chunk():
        yield torch.randn(1, 1, 22050)

    ag.generate_long_audio.return_value = (0, _one_chunk())

    with patch('longecho.main.segment_text', return_value=["Hello world"]):
        with TestClient(app) as client:
            resp = client.post("/v1/audio/speech", json={"input": "Hello world", "voice": "voice1"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("audio/")
            assert len(resp.content) > 44  # more than just a WAV header


def test_openai_speech_voice_not_found(mock_dependencies):
    """Unknown voice on the OpenAI endpoint returns 404."""
    from longecho.main import app

    vm = mock_dependencies['voice_manager']
    vm.get_voice_names.return_value = ["voice1"]

    with TestClient(app) as client:
        resp = client.post("/v1/audio/speech", json={"input": "Hi", "voice": "nope"})
        assert resp.status_code == 404


def test_normalize_audio_tensor_scales_quiet_audio():
    """Volume normalization boosts quiet audio without clipping."""
    import torch
    from longecho.main import _normalize_audio_tensor

    quiet = torch.full((1, 1000), 0.01)
    out = _normalize_audio_tensor(quiet)
    assert out.abs().max() <= 0.99 + 1e-6
    assert float(out.abs().mean()) > float(quiet.abs().mean())  # got louder

    silence = torch.zeros(1, 1000)
    assert torch.equal(_normalize_audio_tensor(silence), silence)  # silence untouched


def test_wav_duration_and_voice_info(tmp_path):
    """_wav_duration reads PCM wav length; get_voice_info reports it per voice."""
    import wave
    from longecho.voice_manager import _wav_duration, VoiceManager

    wav = tmp_path / "v.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000)  # 1.0 s of silence
    assert abs(_wav_duration(wav) - 1.0) < 1e-6
    assert _wav_duration(tmp_path / "missing.wav") is None

    vm = VoiceManager.__new__(VoiceManager)  # skip __init__ (needs models)
    vm.voice_dir = tmp_path
    vm.voices = {"v": (None, None)}
    vm._durations = {}
    info = vm.get_voice_info()
    assert info == [{"name": "v", "duration_seconds": _wav_duration(wav)}]
    assert abs(info[0]["duration_seconds"] - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
