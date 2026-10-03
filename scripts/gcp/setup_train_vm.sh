#!/usr/bin/env bash
# Run ON the training VM after it is created (safe to re-run; each step is skipped when done).
#
#   BUCKET=gs://my-persian-asr bash scripts/gcp/setup_train_vm.sh
#
# 1. /mnt/asr on the persistent boot disk (default) or a RAID-0 of local NVMe SSDs.
# 2. Copy audio + manifests + tokenizer + eval sets from the bucket (incremental).
# 3. Docker + NVIDIA Container Toolkit, the NeMo Speech container
#    (nvcr.io/nvidia/nemo-speech:26.07.00 = NeMo Speech v3.0.0, public on nvcr.io), the matching
#    NeMo source checkout (for examples/ and scripts/), and the base checkpoint.
set -euo pipefail

BUCKET=${BUCKET:?set BUCKET}
IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-speech:26.07.00}
MNT=/mnt/asr

# DATA_ON=boot (default): /mnt/asr is a plain directory on the persistent boot disk. Spot
# preemption wipes local SSDs; on the boot disk the data survives, so a restart re-copies
# nothing. Training reads ~130 MB/s (~1,300 small files/s on 8 GPUs), within pd-balanced
# 1 TB limits, and the 93 GB of audio fits in the page cache.
# DATA_ON=localssd: stripe the bundled local NVMe SSDs (fast, but lost on preemption).
DATA_ON=${DATA_ON:-boot}
if [ "$DATA_ON" = boot ]; then
  sudo mkdir -p "$MNT" && sudo chown "$USER" "$MNT"
elif ! mountpoint -q "$MNT"; then
  mapfile -t ssds < <(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | grep -v -- '-part')
  [ "${#ssds[@]}" -gt 0 ] || { echo "no local NVMe SSDs found" >&2; exit 1; }
  echo "striping ${#ssds[@]} local SSDs -> $MNT"
  sudo mdadm --stop /dev/md0 2>/dev/null || true
  sudo mdadm --create /dev/md0 --level=0 --raid-devices="${#ssds[@]}" "${ssds[@]}" --force --run
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=1,lazy_journal_init=1 /dev/md0
  sudo mkdir -p "$MNT"
  sudo mount -o discard,defaults,nofail /dev/md0 "$MNT"
  sudo chown "$USER" "$MNT"
fi
df -h "$MNT" | tail -1

# Data (rsync is incremental: a no-op when already complete)
for d in audio manifests tokenizer eval; do
  gcloud storage rsync -r "$BUCKET/v1/$d" "$MNT/$d" 2>&1 | tail -1
done

# Docker + NVIDIA Container Toolkit (the DLVM image ships neither). Toolkit
# steps follow NVIDIA's install guide (docs.nvidia.com/datacenter/cloud-native/
# container-toolkit/latest/install-guide.html), version pinned as it shows.
if ! command -v docker >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq docker.io >/dev/null
fi
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq
  V=1.20.1-1
  sudo apt-get install -y -qq nvidia-container-toolkit=$V nvidia-container-toolkit-base=$V \
    libnvidia-container-tools=$V libnvidia-container1=$V >/dev/null
fi
# Configure Docker's nvidia runtime whenever it is missing -- also when the
# image already ships nvidia-ctk (the Deep Learning VM image does).
# (capture first: `docker info | grep -q` can fail under pipefail via SIGPIPE)
docker_info=$(sudo docker info 2>/dev/null || true)
if ! grep -q -i 'nvidia' <<<"$docker_info"; then
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

sudo docker image inspect "$IMAGE" >/dev/null 2>&1 || sudo docker pull -q "$IMAGE"
# The container has only the `nemo` package; examples/ and scripts/ come from
# the matching source tag (NeMo Speech v3.0.0 = commit fd6a877).
[ -d "$HOME/nemo-speech-v3.0.0" ] || git clone -q --depth 1 --branch v3.0.0 \
  https://github.com/NVIDIA-NeMo/Speech.git "$HOME/nemo-speech-v3.0.0"
# Base checkpoint for fine-tuning (train_nemotron.sh reads /models/<this file> inside the container)
mkdir -p "$HOME/models"
[ -f "$HOME/models/nemotron-3.5-asr-streaming-0.6b.nemo" ] || sudo docker run --rm -v "$HOME/models":/models "$IMAGE" \
  python -c "from huggingface_hub import hf_hub_download as d; d('nvidia/nemotron-3.5-asr-streaming-0.6b', 'nemotron-3.5-asr-streaming-0.6b.nemo', local_dir='/models')"
# GPU + NeMo smoke test (fails the script if GPUs or the ASR collection are unusable)
sudo docker run --rm --gpus all "$IMAGE" python -c \
  "import torch, nemo, nemo.collections.asr; assert torch.cuda.device_count() > 0; print('nemo', nemo.__version__, 'torch', torch.__version__, 'gpus', torch.cuda.device_count())"
# Marker for autorun_train.sh: setup is complete on this disk. (With
# DATA_ON=localssd it lives on the SSD array, so it vanishes with the data.)
touch "$MNT/.setup_ok"
