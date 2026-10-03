#!/usr/bin/env bash
# Fine-tune Nemotron 3.5 streaming on Persian, inside the NeMo Speech container
# (run via scripts/gcp/run_in_container.sh, so /opt/nemo-src, /mnt/asr, /models,
# /exp are mounted). Every override below was checked against NeMo Speech
# v3.0.0's examples/asr/conf/fastconformer/cache_aware_streaming/
# fastconformer_transducer_bpe_streaming_prompt.yaml and speech_to_text_finetune.py.
#
#   EXP_NAME=nemotron_fa MAX_STEPS=15000 bash scripts/gcp/run_in_container.sh bash scripts/gcp/train_nemotron.sh
#
# Resumes automatically from the last checkpoint in /exp/$EXP_NAME (Spot preemption).
#
# ++model.char_labels.update_labels=false: speech_to_text_finetune.py (line 129)
# reads cfg.model.char_labels unguarded and this config doesn't define it --
# NVIDIA's tokenizer-extension tutorial passes the same override.
set -euo pipefail

EXP_NAME=${EXP_NAME:?set EXP_NAME}
MAX_STEPS=${MAX_STEPS:?set MAX_STEPS}
DEVICES=${DEVICES:--1}                         # -1 = all visible GPUs
LR=${LR:-1e-4}                                 # NVIDIA guidance: 1e-4 for large fine-tunes
WARMUP=${WARMUP:-$(( MAX_STEPS / 50 ))}        # 2% of max_steps
VAL_EVERY=${VAL_EVERY:-1000}
NUM_SHARDS=$(ls /mnt/asr/shards/train/manifest_*.json | wc -l)

# OOMptimizer profile for this exact model (Nemotron + our tokenizer, fused
# joint) on H100 80 GB, 90% memory, DDP simulated -- see REPRODUCE.md (batch sizes) to re-profile for other GPUs.
BINS='[1.552,1.948,2.569,2.89,3.243,3.639,4.118,4.709,5.397,6.16,6.919,7.819,9.038,10.672,12.715,14.446,15.618,16.421,17.02,17.87,18.49,19.028,19.276,19.52,20.0]'
BATCH='[894,715,572,494,439,390,338,285,240,208,180,152,122,102,82,68,60,56,55,52,51,48,47,44,42]'

exec python /opt/nemo-src/examples/asr/speech_to_text_finetune.py \
  --config-path=/opt/nemo-src/examples/asr/conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming_prompt \
  +init_from_nemo_model=/models/nemotron-3.5-asr-streaming-0.6b.nemo \
  model.tokenizer.dir=/mnt/asr/tokenizer model.tokenizer.type=bpe ++model.tokenizer.update_tokenizer=true \
  ++model.char_labels.update_labels=false \
  model.train_ds.manifest_filepath="/mnt/asr/shards/train/manifest__OP_0..$((NUM_SHARDS - 1))_CL_.json" \
  model.train_ds.is_tarred=false \
  model.train_ds.min_duration=0.1 model.train_ds.max_duration=20.0 \
  model.train_ds.default_prompt_mode=langID \
  ++model.train_ds.use_bucketing=true \
  model.train_ds.batch_duration=null model.train_ds.quadratic_duration=null \
  ++model.train_ds.bucket_duration_bins="$BINS" ++model.train_ds.bucket_batch_size="$BATCH" \
  model.train_ds.num_buckets=25 \
  model.train_ds.bucket_buffer_size=10000 model.train_ds.shuffle_buffer_size=10 \
  model.validation_ds.manifest_filepath=/mnt/asr/manifests/dev.jsonl \
  model.validation_ds.batch_size=16 ++model.validation_ds.default_prompt_mode=langID \
  model.optim.lr="$LR" model.optim.sched.name=CosineAnnealing ~model.optim.sched.d_model \
  model.optim.sched.warmup_steps="$WARMUP" ++model.optim.sched.max_steps="$MAX_STEPS" model.optim.sched.min_lr=1e-6 \
  trainer.devices="$DEVICES" trainer.max_steps="$MAX_STEPS" \
  trainer.limit_train_batches="$VAL_EVERY" trainer.val_check_interval="$VAL_EVERY" \
  trainer.log_every_n_steps=10 \
  exp_manager.exp_dir=/exp exp_manager.name="$EXP_NAME" \
  exp_manager.resume_if_exists=true exp_manager.resume_ignore_no_checkpoint=true \
  exp_manager.checkpoint_callback_params.save_top_k=3
