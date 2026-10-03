# /// script
# requires-python = ">=3.10"
# dependencies = ["gradio==6.29.1", "transformers==5.18.0", "torch", "librosa", "numpy", "soxr"]
# ///
"""Local web app to try the Persian ASR model: live microphone streaming and file transcription.

    uv run scripts/gradio_app.py                          # model from the Hugging Face repo
    uv run scripts/gradio_app.py --model models/hf_transformers   # or a local copy

Then open http://127.0.0.1:7860. It runs only on this machine (no public link).
- Live tab: press record, speak Persian, watch the text appear; press stop to finalize.
- File tab: upload any audio/video file (or record a clip) and transcribe it.

Uses the 🤗 Transformers version of the model, with the same transcription code as
scripts/hf_transcribe_file.py and scripts/hf_stream_mic.py. One model is shared, so requests run
one at a time (a live session holds the model until you press stop).
"""

import argparse
import importlib.util
import threading
import time
import warnings
from pathlib import Path
from threading import Thread

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*max_length.*")

SR = 16000
HERE = Path(__file__).resolve().parent
LOOKAHEAD = {"1.1 s (most accurate)": 13, "0.6 s": 6, "0.3 s (faster)": 3}

spec = importlib.util.spec_from_file_location("hf_file", HERE / "hf_transcribe_file.py")
hf_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_file)  # load_audio, transcribe_whole, transcribe_streaming, pick_device

MODEL = PROCESSOR = None
MODEL_LOCK = threading.Lock()  # one transcription at a time (shared model and processor settings)
FILE_CANCEL = threading.Event()  # set by the File tab's Stop button
LIVE_IDLE_TIMEOUT = 10.0  # s without new audio before a live session is closed (e.g. page reloaded, mic switched)


class LiveSession:
    """One live recording: browser audio in, growing transcript out."""

    def __init__(self, in_sr: int, lookahead: int):
        import soxr
        from transformers import TextIteratorStreamer

        self.resampler = soxr.ResampleStream(in_sr, SR, 1, dtype="float32")
        self.lookahead = lookahead
        self.buf = np.zeros(SR * 60, dtype=np.float32)
        self.n = 0
        self.eof = False
        self.closed = False  # set by finish(); a stream chunk can still arrive after Stop
        self.feed_lock = threading.Lock()
        self.cond = threading.Condition()
        self.text = ""
        self.done = threading.Event()
        self.in_sr = in_sr
        self.last_feed = time.monotonic()
        self.received = 0.0   # seconds of audio received
        self.level_db = -120.0  # loudness of the latest chunk (dBFS)
        self.state = "waiting for the model"
        # group_tokens=False: the tokenizer's decode merges repeated tokens by default (a CTC rule).
        self.streamer = TextIteratorStreamer(PROCESSOR.tokenizer, skip_special_tokens=True, group_tokens=False)
        Thread(target=self._run, daemon=True).start()
        Thread(target=self._collect, daemon=True).start()

    def _append(self, x: np.ndarray) -> None:
        with self.cond:
            while self.n + len(x) > len(self.buf):
                self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)])
            self.buf[self.n:self.n + len(x)] = x
            self.n += len(x)
            self.cond.notify_all()

    def _get(self, start: int, end: int):
        """Samples [start, end); waits for them. None once the session has ended. A session whose
        browser stream just stops (no Stop event: page reloaded, mic switched mid-recording) is
        closed after LIVE_IDLE_TIMEOUT s, so it can't hold the model forever."""
        while True:
            with self.cond:
                if self.n >= end:
                    return self.buf[start:end].copy()
                if self.eof:
                    return None
                self.cond.wait(timeout=1.0)
                idle = time.monotonic() - self.last_feed > LIVE_IDLE_TIMEOUT
            if idle:
                print(f"live: no audio for {LIVE_IDLE_TIMEOUT:.0f} s; closing the session", flush=True)
                self.finish()

    def feed(self, chunk: np.ndarray) -> None:
        with self.feed_lock:
            if self.closed:
                return  # the resampler was already flushed; drop the late chunk
            self.last_feed = time.monotonic()
            self.received += len(chunk) / self.in_sr
            peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
            self.level_db = 20 * np.log10(max(peak, 1e-6))
            self._append(self.resampler.resample_chunk(chunk))

    def finish(self) -> None:
        with self.feed_lock:
            if self.closed:
                return
            self.closed = True
            self._append(self.resampler.resample_chunk(np.zeros(0, np.float32), last=True))
        # two chunks of silence: the model holds back the look-ahead frames of each chunk
        self._append(np.zeros(2 * PROCESSOR.num_samples_per_audio_chunk, np.float32))
        with self.cond:
            self.eof = True
            self.cond.notify_all()

    def _run(self) -> None:
        p, m = PROCESSOR, MODEL
        started = False
        with MODEL_LOCK:
            self.state = "transcribing"
            try:
                p.set_num_lookahead_tokens(self.lookahead)
                first_audio = self._get(0, p.num_samples_first_audio_chunk)
                if first_audio is None:
                    return
                first = p(first_audio, sampling_rate=SR, is_streaming=True, is_first_audio_chunk=True,
                          language="fa-IR", return_tensors="pt").to(m.device, dtype=m.dtype)

                def chunks():
                    yield first.input_features[:, : p.num_mel_frames_first_audio_chunk, :]
                    mel = p.num_mel_frames_first_audio_chunk
                    hop, n_fft = p.feature_extractor.hop_length, p.feature_extractor.n_fft
                    start = mel * hop - n_fft // 2
                    while (x := self._get(start, start + p.num_samples_per_audio_chunk)) is not None:
                        yield p(x, sampling_rate=SR, is_streaming=True, is_first_audio_chunk=False,
                                language="fa-IR", return_tensors="pt").to(m.device, dtype=m.dtype).input_features
                        mel += p.num_mel_frames_per_audio_chunk
                        start = mel * hop - n_fft // 2

                started = True
                m.generate(**{**first, "input_features": chunks()}, streamer=self.streamer)
            finally:
                if not started:
                    self.streamer.end()  # releases the collector thread
                self.state = "finished"
                self.done.set()

    def _collect(self) -> None:
        for piece in self.streamer:
            self.text += piece


def to_float_mono(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    data = data.astype(np.float32)
    return data.mean(axis=1) if data.ndim > 1 else data


def live_status(session) -> str:
    if session is None:
        busy = " · the model is busy with another transcription" if MODEL_LOCK.locked() else ""
        return f"Waiting for audio from the microphone…{busy}"
    level = session.level_db
    hint = " · **very quiet: is the mic muted or the gain too low?**" if level < -50 else ""
    return (f"Audio in: {session.in_sr / 1000:g} kHz · {session.received:.0f} s received · "
            f"level {level:.0f} dB{hint} · {session.state}")


def live_start(lookahead_label, session):
    if session is not None:
        session.finish()  # a previous recording that never got its Stop
    return "", None, live_status(None)


def live_chunk(new_chunk, session, lookahead_label):
    if new_chunk is None:
        return (session.text if session else ""), session, live_status(session)
    sr, data = new_chunk
    if session is None:
        print(f"live: audio from the browser: {sr} Hz, shape {data.shape}, {data.dtype}", flush=True)
        session = LiveSession(sr, LOOKAHEAD[lookahead_label])
    session.feed(to_float_mono(data))
    return session.text, session, live_status(session)


def live_stop(session):
    if session is None:
        return "", None, "No audio was received. Check that the right microphone is selected and not muted."
    session.finish()
    session.done.wait(timeout=30)
    time.sleep(0.2)  # let the collector take the streamer's final text
    return session.text, None, f"Done · {session.received:.0f} s of audio"


def stream_file(audio: np.ndarray):
    """Streaming transcription of a long file; yields (text so far, fraction of audio done).
    Same recipe as hf_transcribe_file.transcribe_streaming, with progress reporting."""
    from transformers import TextIteratorStreamer

    p, m = PROCESSOR, MODEL
    total = len(audio)
    audio = np.concatenate([audio, np.zeros(2 * p.num_samples_per_audio_chunk, np.float32)])  # see that function
    done = [0]
    first = p(audio[: p.num_samples_first_audio_chunk], sampling_rate=SR, is_streaming=True,
              is_first_audio_chunk=True, language="fa-IR", return_tensors="pt").to(m.device, dtype=m.dtype)

    def chunks():
        yield first.input_features[:, : p.num_mel_frames_first_audio_chunk, :]
        mel = p.num_mel_frames_first_audio_chunk
        hop, n_fft = p.feature_extractor.hop_length, p.feature_extractor.n_fft
        start = mel * hop - n_fft // 2
        while (end := start + p.num_samples_per_audio_chunk) < audio.shape[0]:
            if FILE_CANCEL.is_set():
                return  # no more audio: generate finishes, the streamer ends
            done[0] = min(end, total)
            yield p(audio[start:end], sampling_rate=SR, is_streaming=True, is_first_audio_chunk=False,
                    language="fa-IR", return_tensors="pt").to(m.device, dtype=m.dtype).input_features
            mel += p.num_mel_frames_per_audio_chunk
            start = mel * hop - n_fft // 2

    streamer = TextIteratorStreamer(p.tokenizer, skip_special_tokens=True, group_tokens=False, timeout=0.5)
    worker = Thread(target=m.generate, kwargs={**first, "input_features": chunks(), "streamer": streamer})
    worker.start()
    text = ""
    while True:
        try:
            text += next(streamer)
        except StopIteration:
            break
        except Exception:  # queue.Empty: no new text yet; report progress anyway
            pass
        yield text, min(done[0] / total, 0.99)  # 100% only once the last words are out
    worker.join()
    yield text.strip(), 1.0


def transcribe_file(path, lookahead_label):
    """Generator: Gradio shows each yielded (transcript, info) as it comes."""
    if not path:
        yield "", "Upload or record some audio first."
        return
    FILE_CANCEL.clear()
    audio = hf_file.load_audio(Path(path))
    dur = len(audio) / SR
    t = time.perf_counter()
    with MODEL_LOCK:
        PROCESSOR.set_num_lookahead_tokens(LOOKAHEAD[lookahead_label])
        if dur <= 120:
            yield "", f"{dur:.1f} s of audio · transcribing..."
            text = hf_file.transcribe_whole(MODEL, PROCESSOR, audio)
            yield text, f"{dur:.1f} s of audio · whole file · transcribed in {time.perf_counter() - t:.1f} s"
            return
        last, done_frac = 0.0, 0.0
        for text, frac in stream_file(audio):
            if frac < 1.0:
                done_frac = frac
            now = time.perf_counter()
            if frac < 1.0 and now - last < 0.5:
                continue  # refresh the page at most twice a second
            last = now
            el = now - t
            if FILE_CANCEL.is_set() and frac >= 1.0:
                info = f"{dur / 60:.1f} min of audio · **stopped** at {done_frac:.0%} · {el:.0f} s"
            elif frac < 1.0:
                eta = f" · about {el / frac * (1 - frac):.0f} s left" if frac > 0.02 else ""
                info = f"{dur / 60:.1f} min of audio · streaming · **{frac:.0%}** done{eta}"
            else:
                info = f"{dur / 60:.1f} min of audio · streaming · transcribed in {el:.0f} s"
            yield text, info


CSS = ".fa textarea { direction: rtl; text-align: right; font-size: 20px; line-height: 1.9; }"

# Keep a transcript box scrolled to the newest text while it grows (Gradio's autoscroll doesn't act on
# read-only boxes), unless the reader has scrolled up; following resumes at the bottom.
FOLLOW_JS = """(elem_id) => {
  const t = document.querySelector('#' + elem_id + ' textarea');
  if (!t) return;
  if (!t.dataset.followInit) {
    t.dataset.followInit = '1';
    t.dataset.follow = '1';
    t.addEventListener('scroll', () => {
      t.dataset.follow = (t.scrollHeight - t.scrollTop - t.clientHeight < 40) ? '1' : '0';
    });
  }
  if (t.dataset.follow === '1') requestAnimationFrame(() => { t.scrollTop = t.scrollHeight; });
}"""


def follow(box, elem_id):
    import gradio as gr
    box.change(fn=None, js=FOLLOW_JS.replace("(elem_id) =>", "() =>").replace("'#' + elem_id + '", f"'#{elem_id}"))


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Persian ASR") as demo:
        gr.Markdown("## Persian speech recognition · nemotron-asr-streaming-farsi")
        with gr.Tab("Live (microphone)"):
            gr.Markdown("Press **Record**, speak Persian, and the text appears as you talk. Press **Stop** to finalize the last words.")
            la_live = gr.Radio(list(LOOKAHEAD), value=list(LOOKAHEAD)[0], label="Look-ahead (latency)")
            mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
            live_info = gr.Markdown()  # audio level / state, to see at once whether the mic reaches the model
            # fixed height with its own scrollbar; autoscroll keeps the newest words in view
            out_live = gr.Textbox(label="Transcript", lines=12, max_lines=12, elem_id="live_out", elem_classes="fa", rtl=True)
            follow(out_live, "live_out")
            state = gr.State(None)
            mic.start_recording(live_start, inputs=[la_live, state], outputs=[out_live, state, live_info])
            # concurrency_limit=None: a stream left open by an earlier attempt (e.g. while the browser
            # had no mic access) must not block new recordings; MODEL_LOCK still serializes the model.
            mic.stream(live_chunk, inputs=[mic, state, la_live], outputs=[out_live, state, live_info],
                       stream_every=0.25, time_limit=1800, concurrency_limit=None)
            mic.stop_recording(live_stop, inputs=state, outputs=[out_live, state, live_info])
        with gr.Tab("File"):
            gr.Markdown("Upload any audio or video file (mp3, m4a, wav, mp4, ...) or record a clip, then press **Transcribe**.")
            la_file = gr.Radio(list(LOOKAHEAD), value=list(LOOKAHEAD)[0], label="Look-ahead")
            audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
            with gr.Row():
                btn = gr.Button("Transcribe", variant="primary", scale=3)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)
            info_file = gr.Markdown()  # progress line, above the transcript so it stays in view
            out_file = gr.Textbox(label="Transcript", lines=12, max_lines=12, elem_id="file_out", elem_classes="fa", rtl=True)
            follow(out_file, "file_out")
            btn.click(transcribe_file, inputs=[audio_in, la_file], outputs=[out_file, info_file], concurrency_limit=1)
            # queue=False: runs at once, alongside the transcription it stops (long files only; short
            # files finish in a few seconds). The text so far stays.
            stop_btn.click(lambda: FILE_CANCEL.set(), queue=False)
    return demo, CSS


def main() -> None:
    global MODEL, PROCESSOR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=hf_file.DEFAULT_MODEL, help="Hugging Face repo id or local directory")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    a = ap.parse_args()

    from transformers import AutoModelForRNNT, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    device = hf_file.pick_device(a.device)
    print(f"Loading {a.model} ({device})...", flush=True)
    PROCESSOR = AutoProcessor.from_pretrained(a.model)
    MODEL = AutoModelForRNNT.from_pretrained(a.model).to(device).eval()
    demo, css = build_ui()
    demo.queue().launch(server_name="127.0.0.1", server_port=a.port, css=css, inbrowser=not a.no_browser)


if __name__ == "__main__":
    main()
