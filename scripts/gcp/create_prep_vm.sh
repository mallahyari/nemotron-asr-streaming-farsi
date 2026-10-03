#!/usr/bin/env bash
# Create the data-preparation VM: 1x L4 GPU, 32 vCPUs, the source-audio disk
# attached read-only, and a regional bucket for the outputs.
#
#   PROJECT=my-project BUCKET=gs://my-persian-asr DATA_DISK=<disk holding the raw data> bash scripts/gcp/create_prep_vm.sh
#
# Why this shape: cutting and resampling ~1,300 h of audio is CPU-bound (32
# vCPUs), scoring every clip with a Persian ASR model needs a GPU, and a 115M
# model doesn't need more than one L4. The bucket is regional, not zonal, so
# preemptible training VMs in any us-central1 zone can read the result.
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT}
ZONE=${ZONE:-us-central1-a}               # must match the data disk's zone
REGION=${ZONE%-*}
BUCKET=${BUCKET:?set BUCKET, e.g. gs://my-persian-asr}
DATA_DISK=${DATA_DISK:?set DATA_DISK, a disk holding the raw audio + manifests}
VM=${VM:-asr-prep}

gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1 ||
  gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access

# VMs run as the project's default compute service account, which has no
# access to a new bucket until granted (scoped to this bucket only).
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member "serviceAccount:$SA" --role roles/storage.objectAdmin >/dev/null

gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" \
  --machine-type g2-standard-32 \
  --accelerator type=nvidia-l4,count=1 --maintenance-policy TERMINATE \
  --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
  --boot-disk-size 400GB --boot-disk-type pd-balanced \
  --disk "name=$DATA_DISK,mode=ro,device-name=rawdata" \
  --scopes cloud-platform \
  --metadata install-nvidia-driver=True
