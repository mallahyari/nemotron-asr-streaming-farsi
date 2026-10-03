#!/usr/bin/env bash
# Runs on a small always-on supervisor VM as a systemd service (not meant to
# be run by hand).
#
# Spot VMs can't be set to restart automatically ("Spot VMs can't ... be set to
# automatically restart", GCE Spot VM docs), but a preempted (TERMINATED) one
# can be started again whenever capacity exists. Every INTERVAL s: if the
# training VM is stopped, its state object in GCS says "running" (written by
# autorun_train.sh) and no "hold" object exists, start it. A failed start
# (e.g. no H100 capacity in the zone) is simply retried next round.
#
# Config comes from this VM's metadata: train-vm, train-zone, control, interval.
set -uo pipefail
md() { curl -sf -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
VM=$(md train-vm); ZONE=$(md train-zone); CTRL=$(md control); INTERVAL=$(md interval || echo 120)
: "${VM:?}" "${ZONE:?}" "${CTRL:?}"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "watching $VM ($ZONE), control $CTRL, every ${INTERVAL}s"
last=""
while true; do
  status=$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)' 2>&1 | tail -1)
  state=$(gcloud storage cat "$CTRL/state" 2>/dev/null | awk '{print $1}')
  hold=$(gcloud storage ls "$CTRL/hold" 2>/dev/null)
  msg="vm=$status state=${state:-none} hold=${hold:+yes}"
  if [ "$state" = running ] && [ -z "$hold" ] && { [ "$status" = TERMINATED ] || [ "$status" = STOPPED ]; }; then
    # --async: the call needs only compute.instances.start (no operation
    # polling); a capacity failure leaves the VM TERMINATED and we retry.
    if out=$(gcloud compute instances start "$VM" --zone "$ZONE" --async 2>&1); then
      log "$msg -> start requested"
    else
      log "$msg -> start call failed: $(echo "$out" | tail -1)"
    fi
  elif [ "$msg" != "$last" ]; then
    log "$msg"
  fi
  last=$msg
  sleep "$INTERVAL"
done
