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
import re
import sys

# torchcodec ships loaders for FFmpeg 4-8, which link these libavutil sonames.
_TORCHCODEC_AVUTIL = {56, 57, 58, 59, 60}


def _ensure_ffmpeg_on_dll_path() -> None:
    """Make FFmpeg's shared libraries discoverable by torchcodec on Windows.

    torchcodec dynamically loads FFmpeg's DLLs (avutil, avcodec, ...), but since
    Python 3.8 Windows no longer searches PATH for a DLL's *dependencies*. So even
    with FFmpeg's ``bin`` on PATH, ``libtorchcodec_core*.dll`` fails to load with
    "Could not find module ... (or one of its dependencies)". Re-add every PATH
    entry that actually contains FFmpeg DLLs via ``os.add_dll_directory()``, which
    the secure DLL search honors, and warn if the only FFmpeg present is too new
    (9+) for torchcodec. No-op off Windows.
    """
    if sys.platform != "win32":
        return

    found = set()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        directory = entry.strip().strip('"')
        if not directory:
            continue
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        versions = {
            int(m.group(1))
            for name in names
            if (m := re.fullmatch(r"avutil-(\d+)\.dll", name, re.IGNORECASE))
        }
        if not versions:
            continue
        try:
            os.add_dll_directory(directory)
        except OSError:
            continue
        found |= versions

    if found and not (found & _TORCHCODEC_AVUTIL):
        print(
            f"[longecho] Warning: the FFmpeg on your PATH is libavutil {max(found)} "
            "(FFmpeg 9+), but torchcodec supports only FFmpeg 4-8 (libavutil 56-60). "
            "Install an FFmpeg 8 'shared' build (it ships avutil-60.dll) or torchcodec "
            "will fail to load. See the README's Windows FFmpeg section.",
            file=sys.stderr,
        )


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
