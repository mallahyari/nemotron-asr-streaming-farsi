# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers==5.18.0", "torch", "librosa", "numpy"]
# ///
"""Transcribe Persian audio files with the 🤗 Transformers version of the model.

    uv run scripts/hf_transcribe_file.py my_audio.m4a another.mp3
    uv run scripts/hf_transcribe_file.py --lookahead 0.3 my_audio.m4a
    uv run scripts/hf_transcribe_file.py --model models/hf_transformers my_audio.m4a   # local copy

`uv run` reads the dependencies above and builds an isolated environment (the project's own NeMo
environment is not used). The model loads from the Hugging Face repo by default; while it is
private, log in first (`hf auth login`). Any format ffmpeg reads works (mp3, m4a, wav, a video...).

How it transcribes:
- Files up to --max-offline seconds (default 120): one whole-file pass, `model.generate(...)`.
  Memory grows quickly with length: on an M1 Pro (MPS) 64 s took 2.7 s and 4.4 GB, 5 min took
  19 s and 13 GB.
- Longer files: the model's streaming mode, chunk by chunk, so memory stays flat however long the
  file is. Streaming and whole-file transcripts agree to within 0.15 WER points on every test set
  (see README). One difference: Transformers 5.18's streaming only accepts
  full-size chunks, so the end is padded with silence, and a word cut off by the very end of the
  recording can be dropped (whole-file mode keeps it).
The language prompt is always Persian (`fa-IR`).
"""

import argparse
import subprocess
import time
from pathlib import Path
from threading import Thread

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*max_length.*")  # generate's length warning; RNN-T stops by itself

SR = 16000
DEFAULT_MODEL = "mehdi-hf/nemotron-asr-streaming-farsi"
LOOKAHEAD = {"1.1": 13, "0.3": 3, "0.6": 6, "0.08": 0}  # seconds -> right-context frames (80 ms each)


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 with ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def transcribe_whole(model, processor, audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=SR, language="fa-IR").to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True)
    return processor.batch_decode(out.sequences, skip_special_tokens=True)[0]


def transcribe_streaming(model, processor, audio: np.ndarray) -> str:
    """Chunk-by-chunk, as in the model card's streaming recipe."""
    from transformers import TextIteratorStreamer

    # The recipe only processes whole chunks, and the model holds back the last look-ahead frames
    # of each chunk, so the end of the speech would never be transcribed. Two chunks of trailing
    # silence push it through (a chunk already includes the look-ahead).
    audio = np.concatenate([audio, np.zeros(2 * processor.num_samples_per_audio_chunk, dtype=np.float32)])

    first = processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=SR, is_streaming=True,
                      is_first_audio_chunk=True, language="fa-IR", return_tensors="pt").to(model.device, dtype=model.dtype)

    def chunks():
        yield first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel = processor.num_mel_frames_first_audio_chunk
        hop, n_fft = processor.feature_extractor.hop_length, processor.feature_extractor.n_fft
        start = mel * hop - n_fft // 2
        while (end := start + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            x = processor(audio[start:end], sampling_rate=SR, is_streaming=True, is_first_audio_chunk=False,
                          language="fa-IR", return_tensors="pt").to(model.device, dtype=model.dtype)
            yield x.input_features
            mel += processor.num_mel_frames_per_audio_chunk
            start = mel * hop - n_fft // 2

    # group_tokens=False: the tokenizer's decode merges repeated tokens by default (a CTC rule).
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, group_tokens=False)
    th = Thread(target=model.generate, kwargs={**first, "input_features": chunks(), "streamer": streamer})
    th.start()
    text = "".join(streamer)
    th.join()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local directory")
    ap.add_argument("--lookahead", default="1.1", choices=list(LOOKAHEAD), help="seconds the model may hear ahead")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    ap.add_argument("--max-offline", type=float, default=120.0, help="longer files are transcribed in streaming mode")
    a = ap.parse_args()

    from transformers import AutoModelForRNNT, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    device = pick_device(a.device)
    print(f"Loading {a.model} ({device})...", flush=True)
    t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForRNNT.from_pretrained(a.model).to(device).eval()
    processor.set_num_lookahead_tokens(LOOKAHEAD[a.lookahead])
    print(f"Loaded in {time.perf_counter() - t:.0f} s. Look-ahead {a.lookahead} s.\n")

    for f in a.files:
        if not f.is_file():
            print(f"=== {f}: file not found, skipped\n")
            continue
        try:
            audio = load_audio(f)
        except subprocess.CalledProcessError as e:
            print(f"=== {f.name}: ffmpeg could not read it ({e.stderr.decode(errors='replace').strip()[-200:]}), skipped\n")
            continue
        dur = len(audio) / SR
        mode = "whole file" if dur <= a.max_offline else "streaming"
        t = time.perf_counter()
        text = transcribe_whole(model, processor, audio) if mode == "whole file" else transcribe_streaming(model, processor, audio)
        took = time.perf_counter() - t
        print(f"=== {f.name}  ({dur:.1f} s, {mode}, {took:.1f} s to transcribe)")
        print(text + "\n")


if __name__ == "__main__":
    main()
