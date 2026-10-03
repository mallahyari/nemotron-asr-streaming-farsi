#!/usr/bin/env bash
# Run ON the training VM: evaluate a fine-tuned Nemotron .nemo on every eval set
# at two streaming look-aheads, one GPU per (set, look-ahead), all in parallel.
#
#   MODEL=/exp/nemotron_fa/checkpoints/nemotron_fa.nemo NAME=nemotron_fa bash scripts/gcp/eval_nemotron.sh
#
# att_context_size [left, right] in 80 ms frames: [56,3] is the model's default
# (and what in-training val_wer used); [56,13] is its longest look-ahead.
# Results: ~/exp/eval/$NAME/ctx<l>_<r>/<set>/results.json; logs next to them.
# STREAMING=1: true cache-aware streaming inference (chunk by chunk, float32);
# results go to ~/exp/eval/$NAME/stream_ctx<l>_<r>/<set>/.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=${MODEL:?}; NAME=${NAME:?}
STREAMING=${STREAMING:-0}
mode=""; prefix=""
[ "$STREAMING" = 1 ] && { mode="--streaming"; prefix="stream_"; }
SETS=(/mnt/asr/eval/fleurs_test.jsonl /mnt/asr/eval/cv_test.jsonl /mnt/asr/manifests/dev.jsonl /mnt/asr/manifests/test.jsonl)
CTXS=("56,3" "56,13")
# the container runs as root and may have created ~/exp/eval already
sudo mkdir -p "$HOME/exp/eval" && sudo chown "$USER" "$HOME/exp/eval"
gpu=0; pids=()
for ctx in "${CTXS[@]}"; do
  for m in "${SETS[@]}"; do
    set_name=$(basename "$m" .jsonl); tag="${prefix}ctx${ctx/,/_}"
    out="/exp/eval/$NAME/$tag/$set_name"; mkdir -p "$HOME/exp/eval/$NAME/$tag/$set_name"
    log="$HOME/exp/eval/$NAME/$tag/$set_name/eval.log"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/gcp/run_in_container.sh bash -c \
      "PYTHONPATH=/persian-asr:/models/pylib python scripts/evaluate.py --model $MODEL --target-lang fa-IR $mode \
       --nemo-dir /opt/nemo-src --manifest $m --out $out --extra 'att_context_size=[$ctx]'" > "$log" 2>&1 &
    pids+=($!); echo "gpu $gpu: $tag $set_name -> $log"
    gpu=$((gpu + 1))
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "EVAL_EXIT=$rc"
