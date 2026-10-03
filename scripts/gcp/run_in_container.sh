#!/usr/bin/env bash
# Run a command inside the NeMo Speech container on the training VM, with
# NVIDIA's recommended flags (the container warns that the default 64 MB shared
# memory is too small: --ipc=host --ulimit memlock=-1 --ulimit stack=67108864).
#
#   bash scripts/gcp/run_in_container.sh python scripts/evaluate.py ...
#
# Mounts: /mnt/asr (data, same path as in the manifests), ~/models -> /models
# (checkpoints + HF cache), this repo -> /persian-asr (cwd), ~/exp -> /exp, and
# the NeMo Speech v3.0.0 checkout -> /opt/nemo-src (read-only) for its examples/
# and scripts/: the container ships only the `nemo` package, installed from
# /workspace, which is byte-identical to that tag (all 726 .py files, sha1).
set -euo pipefail
IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-speech:26.07.00}
REPO=${REPO:-$HOME/persian-asr}
NEMO_SRC=${NEMO_SRC:-$HOME/nemo-speech-v3.0.0}
mkdir -p "$HOME/models" "$HOME/exp"
# docker run does not inherit the caller's environment: forward the training knobs.
fwd=()
for v in EXP_NAME MAX_STEPS DEVICES LR WARMUP VAL_EVERY CUDA_VISIBLE_DEVICES; do
  [ -n "${!v:-}" ] && fwd+=(-e "$v=${!v}")
done
name=()
[ -n "${CONTAINER_NAME:-}" ] && name=(--name "$CONTAINER_NAME")
exec sudo docker run --rm ${name[@]+"${name[@]}"} --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HOME=/models/hf ${fwd[@]+"${fwd[@]}"} \
  -v /mnt/asr:/mnt/asr -v "$HOME/models":/models -v "$REPO":/persian-asr -v "$HOME/exp":/exp \
  -v "$NEMO_SRC":/opt/nemo-src:ro \
  -w /persian-asr "$IMAGE" "$@"
