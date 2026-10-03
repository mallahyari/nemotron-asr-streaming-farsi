#!/usr/bin/env bash
# Run on your workstation. Creates a small always-on VM that starts the training
# VM again after every spot preemption (scripts/gcp/supervise.sh), and a service
# account for it with only these permissions: start/describe the training VM
# (roles/compute.instanceAdmin.v1 on that one instance, not the project) and
# read the bucket (for the control objects).
#
#   PROJECT=my-project BUCKET=gs://my-persian-asr bash scripts/gcp/create_supervisor.sh
#
# Cost: one e2-micro (the Compute Engine free tier covers one e2-micro in
# us-central1/us-east1/us-west1; otherwise ~US$7/month). Delete after training:
#   gcloud compute instances delete asr-supervisor --zone us-central1-a
# Logs:
#   gcloud compute ssh asr-supervisor --zone us-central1-a --command 'sudo journalctl -u asr-supervisor -n 50'
set -euo pipefail
PROJECT=${PROJECT:?}; BUCKET=${BUCKET:?}
VM=${VM:-asr-train}; ZONE=${ZONE:-us-central1-a}
SUP=${SUP:-asr-supervisor}; SUP_ZONE=${SUP_ZONE:-us-central1-a}
INTERVAL=${INTERVAL:-120}
SA_NAME=asr-supervisor
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
HERE=$(cd "$(dirname "$0")" && pwd)

if ! gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" --display-name "restarts $VM after spot preemption"
  sleep 15   # a new service account can take a few seconds to be usable in IAM bindings
fi
gcloud compute instances add-iam-policy-binding "$VM" --zone "$ZONE" --project "$PROJECT" \
  --member "serviceAccount:$SA" --role roles/compute.instanceAdmin.v1 --format none
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member "serviceAccount:$SA" --role roles/storage.objectViewer --format none

# The startup script runs at every boot of the supervisor VM: it installs the
# loop (passed as metadata) as a systemd service that is restarted if it dies.
startup=$(mktemp)
cat > "$startup" <<'S'
#!/bin/bash
curl -sf -H Metadata-Flavor:Google \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/supervise-script > /usr/local/bin/asr-supervise.sh
chmod +x /usr/local/bin/asr-supervise.sh
cat > /etc/systemd/system/asr-supervisor.service <<U
[Unit]
Description=Restart the spot training VM after preemption
Wants=network-online.target
After=network-online.target
[Service]
ExecStart=/usr/local/bin/asr-supervise.sh
Restart=always
RestartSec=30
[Install]
WantedBy=multi-user.target
U
systemctl daemon-reload
systemctl enable asr-supervisor
systemctl restart asr-supervisor
S
gcloud compute instances create "$SUP" --project "$PROJECT" --zone "$SUP_ZONE" \
  --machine-type e2-micro --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 10GB --boot-disk-type pd-standard \
  --service-account "$SA" --scopes cloud-platform \
  --metadata "train-vm=$VM,train-zone=$ZONE,control=$BUCKET/v1/control/$VM,interval=$INTERVAL" \
  --metadata-from-file "startup-script=$startup,supervise-script=$HERE/supervise.sh"
rm -f "$startup"
