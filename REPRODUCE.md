# Reproducing nemotron-asr-streaming-farsi

These are the steps and commands that produced the released model.

**Setup:**
- Google Cloud with a regional bucket (`gs://<bucket>`) in `us-central1`;
- one L4 VM for data preparation;
- 8× H100 (spot) for training, about 9 hours.

Any CUDA machine works if you adapt `scripts/gcp/`.

**Cost:** the original project, including pilot runs and experiments, cost about $644 on Google Cloud in total.

**Requirements:**
- Python 3.11 or 3.12 (NeMo 3.0 uses 3.11+ syntax), [uv](https://docs.astral.sh/uv/) and `ffmpeg`.
- **Training** runs in the NeMo Speech container `nvcr.io/nvidia/nemo-speech:26.07.00` (NeMo Speech v3.0.0, public, no NGC account needed). `examples/` and `scripts/` come from a matching checkout of [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech) `v3.0.0`.
- **Data preparation** uses this repo's `uv sync --extra train` environment.

## 1. Raw data

There are four openly licensed datasets, about 1,300 h of raw audio. They're downloaded with the [pocket-tts](https://github.com/mallahyari/pocket-tts) scripts onto a persistent disk (about 150 GB):

```bash
git clone https://github.com/mallahyari/pocket-tts && cd pocket-tts && uv sync
uv run python -m training.farsi.prepare_data_fa --sources manatts,filimo,youtube --audio-out /mnt/data/farsi_audio
uv run python training/farsi/v2/ingest_farsi_asr_yt.py --manifest-out /mnt/data/farsi_600h/farsi_asr_yt.jsonl
```

The raw manifests are JSON lines with `path`, `start`, `duration`, `transcript`, `speaker` and `source`:

| File | Rows | Hours | Source |
|---|---|---|---|
| `raw_manatts.jsonl` | 50,058 | 64.5 | Mana-TTS (CC0) |
| `raw_filimo.jsonl` | 402,905 | 248.9 | PerSets Filimo (CC0) |
| `raw_youtube.jsonl` | 303,381 | 296.7 | PerSets YouTube (CC0) |
| `farsi_asr_yt.jsonl` | 163,827 | 694.1 | farsi-asr YouTube (MIT) |

## 2. Prepare the ASR data (L4 VM)

```bash
# workstation: VM with 1x L4, the raw-data disk attached read-only, the bucket created if needed
PROJECT=<project> BUCKET=gs://<bucket> DATA_DISK=<raw-data disk> bash scripts/gcp/create_prep_vm.sh
# on the VM, from a checkout of this repo:
bash scripts/gcp/setup_prep_vm.sh

P="uv run python scripts/prepare_asr_data.py"
$P select   --work-dir /mnt/asr --heldout data/splits/heldout_v1.json \
  --raw /mnt/data/farsi_600h/raw_manatts.jsonl --raw /mnt/data/farsi_600h/raw_filimo.jsonl \
  --raw /mnt/data/farsi_600h/raw_youtube.jsonl --raw /mnt/data/farsi_600h/farsi_asr_yt.jsonl
$P cut      --work-dir /mnt/asr     # cut + resample to 16 kHz mono FLAC
$P score    --work-dir /mnt/asr     # CER of each subtitle vs nvidia/stt_fa_fastconformer_hybrid_large (GPU)
$P finalize --work-dir /mnt/asr --max-cer 0.8 \
  --max-cer-source farsi_asr_yt=0.6 --max-cer-source youtube=0.6 --drop-list data/splits/lid_drops_v1.jsonl
uv run python scripts/verify_manifests.py --work-dir /mnt/asr --heldout data/splits/heldout_v1.json \
  --drop-list data/splits/lid_drops_v1.jsonl --max-cer 0.8 --max-cer-source farsi_asr_yt=0.6 --max-cer-source youtube=0.6
```

`scripts/gcp/run_prep.sh` runs `select → cut → score` unattended, logging to `~/pipeline.log`.

**What the steps do:**
- **`select`** normalizes transcripts to the training form: one spelling per word, standard ZWNJ, numbers as words, no punctuation, no Latin script. It also assigns the frozen held-out split: 30 test and 20 dev speakers per source, never in training.
- **`finalize`** drops two kinds of training clips:
  - clips whose subtitle disagrees badly with the scoring model;
  - English speech found by Whisper language ID: whole videos that are mostly English, and single clips of 3 s or more. These are listed in `data/splits/lid_drops_v1.jsonl`.

  Dev and test aren't CER-filtered.
- **`verify_manifests.py`** re-derives every property from the inputs and fails on any violation.

**Result:**

| Split | Clips | Hours |
|---|---|---|
| train | 782,938 | 1,181.5 |
| dev | 5,791 | 5.2 |
| test | 9,154 | 8.3 |

Two blind listening checks estimated 99.7% usable labels.

## 3. Evaluation sets and tokenizer (L4 VM)

```bash
uv run python scripts/build_eval_sets.py --out-dir /mnt/asr/eval     # FLEURS fa test + Common Voice 22 fa test
uv run python scripts/build_tokenizer.py build --train /mnt/asr/manifests/train.jsonl \
  --check /mnt/asr/manifests/dev.jsonl --check /mnt/asr/manifests/test.jsonl --out-dir /mnt/asr/tokenizer
for d in audio manifests eval tokenizer; do gcloud storage rsync -r /mnt/asr/$d gs://<bucket>/v1/$d; done
```

The tokenizer is Persian-only: SentencePiece unigram, 1,024 pieces, `nmt_nfkc`, with ZWNJ protected as a user-defined symbol. The build verifies that NeMo's tokenizer round-trips every dev and test transcript.

## 4. Train (8× H100 spot)

```bash
# workstation: tries a3-highgpu-8g in us-central1-a/b/c until capacity is found
PROJECT=<project> MACHINES="a3-highgpu-8g" bash scripts/gcp/create_train_vm.sh
# on the VM: data onto the persistent boot disk, Docker + NVIDIA toolkit, container, NeMo source, base model
BUCKET=gs://<bucket> bash scripts/gcp/setup_train_vm.sh
# start the run as a boot-time service (resumes by itself after every restart)
BUCKET=gs://<bucket> EXP_NAME=nemotron_fa MAX_STEPS=15000 VAL_EVERY=500 LR=1e-4 WARMUP=300 \
  bash scripts/gcp/install_autorun.sh
tail -f ~/exp/autorun.log ~/exp/nemotron_fa_*.log
# workstation, optional: a small VM that restarts the training VM after spot preemption
PROJECT=<project> BUCKET=gs://<bucket> bash scripts/gcp/create_supervisor.sh
```

**Recipe** (`scripts/gcp/train_nemotron.sh`):
- `speech_to_text_finetune.py` with NeMo's `fastconformer_transducer_bpe_streaming_prompt` config, initialized from `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- The tokenizer is swapped with `update_tokenizer`. The RNN-T decoder and joint are re-initialized for the new vocabulary; the encoder and the language-prompt layer keep their weights. Training uses the `fa-IR` prompt (index 38) with `prompt_mode=langID`.
- AdamW, LR 1e-4 cosine to 1e-6, 300 warmup steps, 15,000 steps (about 30 passes).
- Lhotse duration bucketing on sharded manifests, clips ≤ 20 s.
- A validation and checkpoint every 500 steps.

**Unattended running** (`autorun_train.sh`):
- **Each boot:** it repairs checkpoints a power loss can leave behind (`fix_checkpoints.py`), then resumes from the last good one.
- **Backup:** it mirrors checkpoints to `gs://<bucket>/v1/exp/` every 10 minutes.
- **End of run:** it powers the VM off when training finishes, or after 3 crashes.
- **Stopping on purpose:** first write `gs://<bucket>/v1/control/<vm>/hold`, or the supervisor will restart the VM.

**Batch sizes** in `train_nemotron.sh` were profiled for H100 80 GB. For other GPUs, re-profile inside the container:

```bash
python /opt/nemo-src/scripts/speech_recognition/estimate_duration_bins.py -b 30 -u 20.0 \
  "/mnt/asr/shards/train/manifest__OP_0..<N-1>_CL_.json"
python scripts/oomptimizer_prompt.py --base /models/nemotron-3.5-asr-streaming-0.6b.nemo \
  --tokenizer-dir /mnt/asr/tokenizer --out /models/nemotron-fa-tokenizer.nemo \
  -- --buckets '<bins from above>' --memory-fraction 0.9 --ddp
```

## 5. Evaluate

```bash
# on the training VM: 4 test sets x 2 look-aheads, one GPU each
MODEL=/exp/nemotron_fa/checkpoints/nemotron_fa.nemo NAME=nemotron_fa bash scripts/gcp/eval_nemotron.sh
STREAMING=1 MODEL=/exp/nemotron_fa/checkpoints/nemotron_fa.nemo NAME=nemotron_fa bash scripts/gcp/eval_nemotron.sh
```

Results land in `~/exp/eval/<NAME>/{ctx,stream_ctx}56_{3,13}/<set>/results.json`.

**The language prompt must be fixed to `fa-IR`.** NeMo 3.0's `speech_to_text_eval.py` / `model.transcribe()` pick this model's prompt at random per utterance, using the untrained `auto` prompt about half the time. `evaluate.py --target-lang fa-IR` runs NeMo's unmodified eval script with the prompt fixed, and checks it covered every utterance. The streaming path (`--streaming`) uses NeMo's cache-aware streaming script with `target_lang=fa-IR`.

**Scoring:** reference and hypothesis both pass through `to_scoring_text`, and WER is cross-checked against NeMo's `word_error_rate`.

**Baseline:**

```bash
uv run python scripts/evaluate.py --model nvidia/stt_fa_fastconformer_hybrid_large --decoder ctc \
  --nemo-dir ~/nemo-speech-v3.0.0 --manifest /mnt/asr/eval/fleurs_test.jsonl --manifest /mnt/asr/manifests/test.jsonl \
  --out /mnt/asr/results/stt_fa_ctc
```

## 6. Use the model

See the [README](README.md#try-it) and the [model card](https://huggingface.co/mehdi-hf/nemotron-asr-streaming-farsi).

The Hugging Face repo has both the `.nemo` and a 🤗 Transformers version (`Nemotron3_5AsrForRNNT`, Transformers ≥ 5.18). The Transformers version is checked against the `.nemo`: bit-identical weights and the same FLEURS WER.
