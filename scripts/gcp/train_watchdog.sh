#!/usr/bin/env bash
# Run ON the training VM: launch train_nemotron.sh in a named container and
# stop that container if it hangs after training has finished.
#
#   EXP_NAME=nemotron_fa MAX_STEPS=15000 VAL_EVERY=500 bash scripts/gcp/train_watchdog.sh
#
# Observed in practice: after "`Trainer.fit` stopped" and the final .nemo save,
# the process (DDP/dataloader teardown) never exits on its own. On paid GPUs
# that idles until someone notices. Once the log says fit stopped, we wait
# until the log has been quiet for QUIET seconds (checkpoint writes done),
# then stop the container. The exit code is appended as TRAIN_EXIT=<rc>.
# Exits 0 iff training finished (see the end of this file).
set -uo pipefail
cd "$(dirname "$0")/../.."
EXP_NAME=${EXP_NAME:?set EXP_NAME}
QUIET=${QUIET:-60}
export CONTAINER_NAME="train-${EXP_NAME//[^a-zA-Z0-9_.-]/_}"
LOG="$HOME/exp/${EXP_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$HOME/exp"
sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
# If the watchdog itself is interrupted, don't leave the GPU container running.
trap 'echo "watchdog: interrupted; stopping container $CONTAINER_NAME" >> "$LOG"; sudo docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1; exit 130' INT TERM

bash scripts/gcp/run_in_container.sh ${TRAIN_CMD:-bash scripts/gcp/train_nemotron.sh} > "$LOG" 2>&1 &
pid=$!
echo "training pid $pid, container $CONTAINER_NAME, log $LOG"

finished=0
while kill -0 "$pid" 2>/dev/null; do
  if [ "$finished" = 0 ] && grep -a -q 'Trainer.fit. stopped: .max_steps=' "$LOG"; then
    finished=1
  fi
  if [ "$finished" = 1 ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ "$age" -ge "$QUIET" ]; then
      echo "watchdog: fit finished and log quiet for ${age}s; stopping container $CONTAINER_NAME" >> "$LOG"
      sudo docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
      break
    fi
  fi
  sleep 10
done
wait "$pid"; rc=$?
[ "$finished" = 1 ] && rc_note=" (training finished; container stopped by watchdog if still running)" || rc_note=""
echo "TRAIN_EXIT=$rc$rc_note" >> "$LOG"
# Exit 0 only if training reached the end (Lightning's fit_loop prints
# "`Trainer.fit` stopped: `max_steps=N` reached."; other stop reasons such as
# "No training batches." don't count); otherwise
# the training exit code (3 if it exited 0 without finishing). autorun_train.sh
# relies on this to tell "done" from "crashed".
[ "$finished" = 1 ] && exit 0
[ "$rc" = 0 ] && exit 3
exit "$rc"
