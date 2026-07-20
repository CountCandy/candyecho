# LongEcho: Long-form audio generation with Echo-TTS
"""
LongEcho package for generating long-form audio using Echo-TTS.

This package provides:
- Text segmentation for chunking long text
- Voice management with caching
- Audio generation with continuation for seamless long-form output
- FastAPI server for web interface
"""

import os
import sys


def _ensure_ffmpeg_on_dll_path() -> None:
    """Make FFmpeg's shared libraries discoverable by torchcodec on Windows.

    torchcodec dynamically loads FFmpeg's DLLs (avutil, avcodec, ...), but since
    Python 3.8 Windows no longer searches PATH for a DLL's *dependencies*. So even
    with FFmpeg's ``bin`` on PATH, ``libtorchcodec_core*.dll`` fails to load with
    "Could not find module ... (or one of its dependencies)". Re-add every PATH
    entry that actually contains FFmpeg DLLs via ``os.add_dll_directory()``, which
    the secure DLL search honors. No-op off Windows.
    """
    if sys.platform != "win32":
        return
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        directory = entry.strip().strip('"')
        if not directory:
            continue
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        if any(n.lower().startswith("avutil") and n.lower().endswith(".dll") for n in names):
            try:
                os.add_dll_directory(directory)
            except OSError:
                continue


# Must run before any submodule imports torchcodec (via the vendored echo_tts).
_ensure_ffmpeg_on_dll_path()

# Eager imports: modules with no external dependencies
from .text_segmenter import segment_text  # noqa: E402
from .text_normalizer import TextNormalizer  # noqa: E402

__all__ = [
    "segment_text",
    "TextNormalizer",
    "VoiceManager",
    "AudioGenerator",
]

# Lazy imports for modules that depend on external 'inference' package (PEP 562)
_lazy_imports = {
    "VoiceManager": ".voice_manager",
    "AudioGenerator": ".audio_generator",
}


def __getattr__(name: str):
    """Lazily import VoiceManager and AudioGenerator to defer 'inference' dependency."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
