#!/usr/bin/env bash
# Run ON the prep VM once: mount the raw-audio disk read-only, install ffmpeg,
# uv and this repo's training extras (NeMo).
#
#   bash scripts/gcp/setup_prep_vm.sh            # from the repo checkout on the VM
set -euo pipefail

sudo mkdir -p /mnt/data /mnt/asr
mountpoint -q /mnt/data || sudo mount -o ro,noload /dev/disk/by-id/google-rawdata /mnt/data
sudo chown "$USER" /mnt/asr

sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg libsndfile1 >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --extra train
uv run python -c "import torch, nemo; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'nemo', nemo.__version__)"
