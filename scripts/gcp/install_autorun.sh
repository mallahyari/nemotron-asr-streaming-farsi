#!/usr/bin/env bash
# Run ON the training VM (as your normal user) to make an experiment run
# unattended and survive spot preemption. Writes ~/train.env and installs the
# asr-train systemd service, which runs scripts/gcp/autorun_train.sh at every
# boot. Starts it now unless NO_START=1.
#
#   BUCKET=gs://my-persian-asr EXP_NAME=full MAX_STEPS=15000 VAL_EVERY=500 LR=1e-4 WARMUP=300 \
#     bash scripts/gcp/install_autorun.sh
#
# Optional: DEVICES SYNC_EVERY (600 s) MAX_CRASHES (3) AUTO_POWEROFF (1).
# Watch: tail -f ~/exp/autorun.log (service), ~/exp/<EXP_NAME>_*.log (training).
# Remove: sudo systemctl disable --now asr-train
#
# Why systemd and not tmux: preemption stops the whole VM, so a tmux session
# dies with it; a systemd service starts again by itself when the VM boots.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
: "${BUCKET:?}" "${EXP_NAME:?}" "${MAX_STEPS:?}"
case "$EXP_NAME" in *[!a-zA-Z0-9_.-]*) echo "EXP_NAME: use only letters, digits, _ . -" >&2; exit 1;; esac

{
  echo "# written by install_autorun.sh $(date -u +%FT%TZ)"
  for v in BUCKET EXP_NAME MAX_STEPS VAL_EVERY LR WARMUP DEVICES SYNC_EVERY MAX_CRASHES AUTO_POWEROFF; do
    [ -n "${!v:-}" ] && printf '%s=%q\n' "$v" "${!v}"
  done
} > "$HOME/train.env"
cat "$HOME/train.env"
mkdir -p "$HOME/exp"

# After=docker.service also orders shutdown: this service (and its watchdog's
# `docker stop`) is stopped before Docker.
sudo tee /etc/systemd/system/asr-train.service >/dev/null <<UNIT
[Unit]
Description=Persian ASR training (resumes after spot preemption)
Wants=network-online.target
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO
Environment=PATH=/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/bin/bash $REPO/scripts/gcp/autorun_train.sh
Restart=no
KillMode=control-group
TimeoutStopSec=60
StandardOutput=append:$HOME/exp/autorun.log
StandardError=append:$HOME/exp/autorun.log

[Install]
WantedBy=multi-user.target
UNIT
# The VM's gcloud is a snap, and snap runs it in a scope under the user's
# systemd manager (user@UID.service). Without lingering, that manager is
# stopped when the last SSH session logs out, killing any gcloud upload in
# flight (checkpoint syncs otherwise die with "Terminated"). Lingering
# keeps the user manager running permanently.
sudo loginctl enable-linger "$USER"
sudo systemctl daemon-reload
sudo systemctl enable asr-train
if [ "${NO_START:-0}" != 1 ]; then
  sudo systemctl restart asr-train
  sleep 3
  systemctl --no-pager status asr-train | head -5
fi
