# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers==5.18.0", "torch", "librosa", "numpy"]
# ///
"""Live Persian transcription from the microphone with the 🤗 Transformers version of the model.

    uv run scripts/hf_stream_mic.py                     # speak Persian; Ctrl+C to stop
    uv run scripts/hf_stream_mic.py --lookahead 0.3     # lower latency, slightly less accurate
    uv run scripts/hf_stream_mic.py --seconds 20        # stop by itself after 20 s
    uv run scripts/hf_stream_mic.py --simulate some_recording.flac
        # play a file in at real-time speed instead of the mic (same code path; for testing)

macOS asks for microphone permission for your terminal app the first time. The mic is read with
ffmpeg (`-f avfoundation`); `--mic` picks the input device (list them with
`ffmpeg -f avfoundation -list_devices true -i ""`). The model loads from the Hugging Face repo by
default (log in with `hf auth login` while it is private) or from a local directory via --model.

Text appears as the model commits to it. With the 1.1 s look-ahead each word shows up roughly
1-1.5 s after it is spoken; with 0.3 s, about half a second. When you stop, two chunks of silence are
fed in so the last words are finalized.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from threading import Thread

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*max_length.*")  # generate's length warning; RNN-T stops by itself

SR = 16000
DEFAULT_MODEL = "mehdi-hf/nemotron-asr-streaming-farsi"
LOOKAHEAD = {"1.1": 13, "0.3": 3, "0.6": 6, "0.08": 0}


class LiveAudio:
    """Growing buffer of 16 kHz float32 samples filled by an ffmpeg subprocess."""

    def __init__(self, cmd: list[str], tail_samples: int):
        self.tail_samples = tail_samples  # silence appended at the end so the last words are emitted
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        self.buf = np.zeros(SR * 60, dtype=np.float32)
        self.n = 0
        self.eof = False
        self.cond = threading.Condition()
        Thread(target=self._read, daemon=True).start()

    def _append(self, x: np.ndarray) -> None:
        with self.cond:
            while self.n + len(x) > len(self.buf):
                self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)])
            self.buf[self.n:self.n + len(x)] = x
            self.n += len(x)
            self.cond.notify_all()

    def _read(self) -> None:
        pending = b""
        while True:
            data = self.proc.stdout.read(SR // 10 * 4)  # ~100 ms per read
            if not data:
                break
            pending += data
            k = len(pending) // 4 * 4
            self._append(np.frombuffer(pending[:k], dtype=np.float32).copy())
            pending = pending[k:]
        self._append(np.zeros(self.tail_samples, dtype=np.float32))
        with self.cond:
            self.eof = True
            self.cond.notify_all()

    def get(self, start: int, end: int) -> np.ndarray | None:
        """Samples [start, end); waits for them. None if the stream ended first."""
        with self.cond:
            while self.n < end and not self.eof:
                self.cond.wait()
            if self.n < end:
                return None
            return self.buf[start:end].copy()

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)  # ffmpeg flushes and exits; the reader then sees EOF


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local directory")
    ap.add_argument("--lookahead", default="1.1", choices=list(LOOKAHEAD), help="seconds the model may hear ahead")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    ap.add_argument("--mic", default="0", help="avfoundation audio device index")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--simulate", default=None, help="stream this file at real-time speed instead of the mic")
    a = ap.parse_args()

    from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    device = pick_device(a.device)
    print(f"Loading {a.model} ({device})...", flush=True)
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForRNNT.from_pretrained(a.model).to(device).eval()
    processor.set_num_lookahead_tokens(LOOKAHEAD[a.lookahead])
    # warm up the GPU kernels so the first words aren't delayed
    warm = processor(np.zeros(SR, np.float32), sampling_rate=SR, language="fa-IR").to(device, dtype=model.dtype)
    with torch.inference_mode():
        model.generate(**warm)

    out = ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    if a.simulate:
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-re", "-i", a.simulate, *out]
    else:
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-f", "avfoundation", "-i", f":{a.mic}", *out]
    if a.seconds:
        cmd[-len(out):-len(out)] = ["-t", str(a.seconds)]
    # two chunks of silence at the end: the model holds back the look-ahead frames of each chunk
    src = LiveAudio(cmd, tail_samples=2 * processor.num_samples_per_audio_chunk)
    what = f"file {a.simulate} (real-time)" if a.simulate else f"microphone {a.mic}"
    print(f"Listening to {what}. Look-ahead {a.lookahead} s. Speak Persian; Ctrl+C to stop.\n", flush=True)

    first_audio = src.get(0, processor.num_samples_first_audio_chunk)
    if first_audio is None:
        sys.exit("No audio received. Check microphone permission for your terminal (System Settings > "
                 "Privacy & Security > Microphone) or pick another device with --mic.")
    first = processor(first_audio, sampling_rate=SR, is_streaming=True, is_first_audio_chunk=True,
                      language="fa-IR", return_tensors="pt").to(device, dtype=model.dtype)

    def chunks():
        yield first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel = processor.num_mel_frames_first_audio_chunk
        hop, n_fft = processor.feature_extractor.hop_length, processor.feature_extractor.n_fft
        start = mel * hop - n_fft // 2
        while (x := src.get(start, start + processor.num_samples_per_audio_chunk)) is not None:
            feats = processor(x, sampling_rate=SR, is_streaming=True, is_first_audio_chunk=False,
                              language="fa-IR", return_tensors="pt").to(device, dtype=model.dtype)
            yield feats.input_features
            mel += processor.num_mel_frames_per_audio_chunk
            start = mel * hop - n_fft // 2

    # group_tokens=False: the tokenizer's decode merges repeated tokens by default (a CTC rule).
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, group_tokens=False)
    worker = Thread(target=model.generate, kwargs={**first, "input_features": chunks(), "streamer": streamer})
    worker.start()
    # First Ctrl+C: stop the audio; the last words are still finalized, then the stream ends by
    # itself. Second Ctrl+C: quit at once. (A handler, because the main thread spends its time
    # blocked waiting on the streamer, where a KeyboardInterrupt isn't delivered promptly.)
    stopping = threading.Event()

    def on_sigint(signum, frame):
        if stopping.is_set():
            print("\n[quit]", flush=True)
            os._exit(130)
        stopping.set()
        print("\n[stopping: finishing the last words...]", flush=True)
        src.stop()

    signal.signal(signal.SIGINT, on_sigint)
    t0 = time.perf_counter()
    for text in streamer:
        print(text, end="", flush=True)
    worker.join()
    src.stop()
    print(f"\n\n[done after {time.perf_counter() - t0:.0f} s]")


if __name__ == "__main__":
    main()
