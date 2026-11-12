#!/usr/bin/env bash
# One-time setup for the optional engine (Fish Audio Drama 3).
#
# Both are Linux-only, so on Windows run this inside WSL:
#
#     wsl
#     cd /mnt/c/path/to/candyecho
#     bash scripts/setup_engines.sh
#
# It installs each engine's code and Python environment under
# ~/candyecho-engines, which is where CandyEcho looks when you press "Start".
# Model weights are NOT downloaded here: use the Download button in the web UI,
# which stores them in CandyEcho's models/ folder.
#
# Safe to re-run: existing checkouts are updated, environments are re-synced.
set -euo pipefail

ENGINE_HOME="$HOME/candyecho-engines"
FISH_REPO="https://github.com/fishaudio/fish-speech.git"
# PyTorch build for Drama 3. cu129 covers current cards including RTX 50-series
# and RTX PRO Blackwell; use cu126 only for an old driver.
FISH_CUDA_EXTRA="${FISH_CUDA_EXTRA:-cu129}"

if [ "$#" -eq 0 ]; then set -- drama3; fi
for engine in "$@"; do
    case "$engine" in
        drama3) ;;
        *) echo "Unknown engine '$engine' (expected: drama3)" >&2; exit 2 ;;
    esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [ "$(uname -s)" != "Linux" ]; then
    echo "This script must run on Linux (inside WSL on Windows)." >&2
    exit 1
fi

say "Checking that the GPU is visible"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi does not work here. On Windows, update the NVIDIA driver and" >&2
    echo "run 'wsl --update' from PowerShell, then try again." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

say "Installing system packages (asks for your Linux password)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential ffmpeg portaudio19-dev libsox-dev

if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$ENGINE_HOME"

checkout() {  # checkout <repo-url> <dir>
    if [ -d "$2/.git" ]; then
        git -C "$2" pull --ff-only || echo "  (could not update $2; keeping the existing checkout)"
    else
        git clone --depth 1 "$1" "$2"
    fi
}

for engine in "$@"; do
    case "$engine" in
        drama3)
            say "Setting up Fish Audio Drama 3 (fish-speech)"
            checkout "$FISH_REPO" "$ENGINE_HOME/fish-speech"
            (cd "$ENGINE_HOME/fish-speech" && uv sync --python 3.12 --extra "$FISH_CUDA_EXTRA")
            ;;
    esac
done

say "Done"
echo "Engines are installed in $ENGINE_HOME."
echo "Now open CandyEcho, pick the engine, press Download, and it will start by itself."
