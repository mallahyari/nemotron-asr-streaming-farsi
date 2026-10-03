#!/usr/bin/env bash
# Run ON the prep VM: select -> cut -> score, logging to ~/pipeline.log.
# Stops before `finalize`, whose CER cutoff is chosen after looking at the
# score distribution (REPRODUCE.md).
#
#   nohup setsid bash scripts/gcp/run_prep.sh >/dev/null 2>&1 < /dev/null &
#   tail -f ~/pipeline.log
set -uo pipefail

cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
RAW=${RAW:-/mnt/data/farsi_600h}
WORK=${WORK:-/mnt/asr}
LOG=${LOG:-$HOME/pipeline.log}
P="uv run python scripts/prepare_asr_data.py"

{
  echo "### select $(date -u)" &&
    $P select --work-dir "$WORK" \
      --raw "$RAW/raw_manatts.jsonl" --raw "$RAW/raw_filimo.jsonl" \
      --raw "$RAW/raw_youtube.jsonl" --raw "$RAW/farsi_asr_yt.jsonl" &&
    echo "### cut $(date -u)" && $P cut --work-dir "$WORK" &&
    echo "### score $(date -u)" && $P score --work-dir "$WORK" &&
    echo "### DONE $(date -u)" || echo "### FAILED $(date -u)"
} > "$LOG" 2>&1
