#!/usr/bin/env bash
# Create the training VM: an A3 High H100 machine as a Spot VM, trying each
# machine size in MACHINES (largest first) in each zone until one has capacity.
#
#   PROJECT=my-project bash scripts/gcp/create_train_vm.sh                  # 8 GPUs
#   PROJECT=my-project MACHINES="a3-highgpu-4g a3-highgpu-2g" bash ...     # smaller machines
#
# - SPOT provisioning: our H100 quota is preemptible-only. On preemption the
#   VM STOPs (boot disk kept); local-SSD contents are lost, so setup_train_vm.sh
#   re-copies data from the bucket and checkpoints live on the boot disk + GCS.
# - --maintenance-policy=TERMINATE: required for GPU VMs (no live migration).
# - a3-highgpu-8g bundles 16 local NVMe SSDs (~6 TB); setup_train_vm.sh stripes
#   them into /mnt/asr, the path our manifests use.
# - Boot disk holds the NeMo container image, HF cache and experiment dirs.
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT}
VM=${VM:-asr-train}
ZONES=${ZONES:-"us-central1-a us-central1-b us-central1-c"}
MACHINES=${MACHINES:-a3-highgpu-8g}
BOOT_GB=${BOOT_GB:-1000}

for machine in $MACHINES; do
for zone in $ZONES; do
  echo "== trying $machine in $zone"
  if gcloud compute instances create "$VM" --project "$PROJECT" --zone "$zone" \
      --machine-type "$machine" \
      --provisioning-model SPOT --instance-termination-action STOP \
      --maintenance-policy TERMINATE \
      --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
      --boot-disk-size "${BOOT_GB}GB" --boot-disk-type pd-balanced \
      --scopes cloud-platform \
      --metadata install-nvidia-driver=True; then
    # A STOCKOUT can also surface after creation starts: confirm it is running.
    sleep 20
    status=$(gcloud compute instances describe "$VM" --project "$PROJECT" --zone "$zone" --format='value(status)' 2>/dev/null || true)
    if [ "$status" = "RUNNING" ]; then
      echo "created $VM ($machine) in $zone"
      exit 0
    fi
    echo "  $VM in $zone is '$status', not RUNNING; trying next" >&2
    gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$zone" --quiet >/dev/null 2>&1 || true
  fi
done
done
echo "no capacity for [$MACHINES] in [$ZONES]" >&2
exit 1
