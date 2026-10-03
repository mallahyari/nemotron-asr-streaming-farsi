#!/usr/bin/env bash
# Unattended, preemption-proof training on a spot VM. Started at every boot by
# the asr-train systemd service (installed by scripts/gcp/install_autorun.sh);
# not meant to be run by hand. Config comes from ~/train.env.
#
# Each boot (first start, or restart after a preemption):
#   1. If this experiment already ended (done/failed, recorded in ~/exp): exit.
#   2. Setup, once per disk (marker /mnt/asr/.setup_ok): data is on the
#      persistent boot disk, so a restart re-copies nothing.
#   3. Shards, once: built into a temp dir and renamed, so an interrupted build
#      is simply redone.
#   4. fix_checkpoints.py: clear the two preemption states NeMo can't resume
#      from (unfinished last checkpoint; two finished last checkpoints).
#   5. train_watchdog.sh: NeMo resumes from the last checkpoint
#      (exp_manager.resume_if_exists). Meanwhile: `sync` every FLUSH_EVERY s,
#      so a finished checkpoint is on disk, not just in the page cache (in our
#      preemption test a "finished" 7.5 GB last.ckpt came back truncated after
#      the power cut); and ~/exp/$EXP_NAME is mirrored to GCS every SYNC_EVERY s,
#      as a backup in case the disk itself is lost.
#   6. Finished: final sync, state "done" in GCS, power off (stops billing).
#      Crashed (not preempted): retry; after MAX_CRASHES crashes, state
#      "failed" and power off, so a broken run can't burn GPU hours.
# A preemption kills all of this mid-way; that's fine. The supervisor VM
# (scripts/gcp/create_supervisor.sh) starts this VM again while the state in
# GCS is "running" and no "hold" object exists, and we're back at step 1.
set -uo pipefail
cd "$(dirname "$0")/../.."
source "$HOME/train.env"
: "${EXP_NAME:?}" "${MAX_STEPS:?}" "${BUCKET:?}"
export EXP_NAME MAX_STEPS LR WARMUP VAL_EVERY DEVICES   # unset ones stay unset
SYNC_EVERY=${SYNC_EVERY:-600}
FLUSH_EVERY=${FLUSH_EVERY:-15}
MAX_CRASHES=${MAX_CRASHES:-3}
AUTO_POWEROFF=${AUTO_POWEROFF:-1}
VM_NAME=${VM_NAME:-$(hostname)}
CTRL="$BUCKET/v1/control/$VM_NAME"          # read by the supervisor
GCS_EXP="$BUCKET/v1/exp/$EXP_NAME"
EXP="$HOME/exp/$EXP_NAME"
STATE="$HOME/exp/$EXP_NAME.state"            # local: done | failed
CRASHES="$HOME/exp/$EXP_NAME.crashes"

log() { echo "$(date -u +%FT%TZ) autorun[$EXP_NAME]: $*"; }

set_state() {  # retry: the supervisor acts on this
  for _ in 1 2 3 4 5; do
    echo "$1 $EXP_NAME $(date -u +%FT%TZ)" | gcloud storage cp - "$CTRL/state" --quiet 2>/dev/null && return 0
    sleep 10
  done
  log "WARNING: could not write $CTRL/state=$1"
}

sync_exp() {
  # Never mirror a dir without a finished last checkpoint: --delete-unmatched
  # would otherwise wipe the GCS backup if the local copy were ever empty.
  if ! ls "$EXP"/checkpoints/*last.ckpt >/dev/null 2>&1; then
    log "sync skipped: no last checkpoint yet"; return 0
  fi
  gcloud storage rsync -r --delete-unmatched-destination-objects "$EXP" "$GCS_EXP/run" --quiet \
    && gcloud storage cp "$HOME/exp/${EXP_NAME}"_*.log "$GCS_EXP/logs/" --quiet \
    && log "synced to $GCS_EXP" || log "WARNING: sync failed (retrying next round)"
}

finish() {  # $1 = done | failed
  echo "$1" > "$STATE"
  sync_exp
  set_state "$1"
  log "experiment $1"
  if [ "$AUTO_POWEROFF" = 1 ]; then log "powering off"; sudo systemctl poweroff; fi
  exit 0
}

wd_pid=""; sync_pid=""; flush_pid=""
on_term() {  # VM shutdown / preemption: flush to disk, then get out
  log "SIGTERM (shutdown or preemption); flushing and stopping"
  sync
  [ -n "$flush_pid" ] && kill "$flush_pid" 2>/dev/null
  [ -n "$wd_pid" ] && kill -TERM "$wd_pid" 2>/dev/null
  [ -n "$sync_pid" ] && { pkill -P "$sync_pid"; kill "$sync_pid"; } 2>/dev/null
  wait
  exit 0
}
trap on_term TERM INT

if [ -f "$STATE" ]; then
  s=$(cat "$STATE"); log "already $s; nothing to do"
  set_state "$s"   # in case the earlier write didn't land
  exit 0
fi
set_state running
log "boot: start (MAX_STEPS=$MAX_STEPS VAL_EVERY=${VAL_EVERY:-default} LR=${LR:-default})"

crashes=$(cat "$CRASHES" 2>/dev/null || echo 0)
while true; do
  if [ "$crashes" -ge "$MAX_CRASHES" ]; then
    log "$crashes crashes; giving up"; finish failed
  fi

  if [ ! -f /mnt/asr/.setup_ok ]; then
    log "setup"
    BUCKET="$BUCKET" bash scripts/gcp/setup_train_vm.sh & wait $! || { log "setup failed"; crashes=$((crashes + 1)); echo "$crashes" > "$CRASHES"; sleep 60; continue; }
  fi

  if [ ! -f /mnt/asr/shards/train/.complete ]; then
    log "building shards"
    sudo rm -rf /mnt/asr/shards/train /mnt/asr/shards/train.tmp
    if bash scripts/gcp/run_in_container.sh python scripts/shard_manifest.py \
         /mnt/asr/manifests/train.jsonl /mnt/asr/shards/train.tmp --per-shard 200 & wait $!; then
      sudo touch /mnt/asr/shards/train.tmp/.complete && sudo mv /mnt/asr/shards/train.tmp /mnt/asr/shards/train
    else
      log "sharding failed"; crashes=$((crashes + 1)); echo "$crashes" > "$CRASHES"; sleep 60; continue
    fi
  fi

  sudo python3 scripts/gcp/fix_checkpoints.py "$EXP/checkpoints"

  ( while sleep "$FLUSH_EVERY"; do sync; done ) &
  flush_pid=$!
  ( while sleep "$SYNC_EVERY"; do sync_exp; done ) &
  sync_pid=$!
  log "training (attempt $((crashes + 1)))"
  bash scripts/gcp/train_watchdog.sh &
  wd_pid=$!
  wait "$wd_pid"; rc=$?
  wd_pid=""
  { pkill -P "$sync_pid"; kill "$sync_pid" "$flush_pid"; wait "$sync_pid" "$flush_pid"; } 2>/dev/null; sync_pid=""; flush_pid=""
  sync

  if [ "$rc" = 0 ]; then
    finish done
  fi
  crashes=$((crashes + 1)); echo "$crashes" > "$CRASHES"
  log "training exited rc=$rc without finishing (crash $crashes/$MAX_CRASHES)"
  sync_exp
  sleep 60
done
