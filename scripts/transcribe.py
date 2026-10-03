"""Transcribe Persian speech with the fine-tuned Nemotron streaming model, chunk by chunk.

    # your own audio files (any format ffmpeg reads: mp3, m4a, wav, ogg, a video...)
    uv run --extra train python scripts/transcribe.py my_audio.m4a another.mp3

    # a NeMo manifest with reference transcripts: also prints WER
    uv run --extra train python scripts/transcribe.py --manifest my_clips.jsonl

    # record N seconds from the microphone, then transcribe (macOS asks for mic access)
    uv run --extra train python scripts/transcribe.py --record 8

    # watch the text appear at real speaking speed, as in a live stream
    uv run --extra train python scripts/transcribe.py --realtime my_audio.m4a

The model (`nemotron-asr-streaming-farsi.nemo`) is downloaded from the Hugging Face repo on first
use and cached; `--model path/to/file.nemo` uses a local copy instead.

Inference is the same as NeMo's cache-aware streaming script
(examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py),
which scored FLEURS 8.8% / conversational test 26.0% WER (README):
- the `fa-IR` language prompt via `set_inference_prompt`,
- `CacheAwareStreamingAudioBuffer` + `conformer_stream_step`, one chunk at a time,
- greedy RNNT decoding, float32 (cache-aware streaming doesn't support bf16).
Picks the GPU automatically (Apple Silicon `mps`, or `cuda`), else the CPU. On an
M1 Pro, MPS gave the same transcripts as the CPU and ran ~10x faster than real time;
the CPU runs about real time.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
from typing_extensions import Annotated

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
HF_REPO, HF_FILE = "mehdi-hf/nemotron-asr-streaming-farsi", "nemotron-asr-streaming-farsi.nemo"
SAMPLE_RATE = 16000
CONTEXTS = {"1.1": [56, 13], "0.3": [56, 3]}  # look-ahead seconds -> att_context_size [left, right]

app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 with ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def record(seconds: float, mic: str) -> Path:
    out_dir = REPO / "models" / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rec_{datetime.now():%Y%m%d_%H%M%S}.wav"
    print(f"Recording {seconds:g} s from microphone {mic!r}... speak Persian now.", flush=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "avfoundation", "-i", f":{mic}", "-t", str(seconds),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), str(out)], check=True)
    print(f"Saved {out}")
    return out


def quiet_nemo() -> None:
    import logging as pylogging
    import warnings

    warnings.filterwarnings("ignore")
    for name in ("nemo_logger", "lightning", "lightning.pytorch", "numba"):
        pylogging.getLogger(name).setLevel(pylogging.ERROR)
    from nemo.utils import logging as nemo_logging
    nemo_logging.setLevel(nemo_logging.ERROR)


def load_model(path: Path, context: list[int], device: str):
    import torch
    from omegaconf import OmegaConf
    quiet_nemo()
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

    if path is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(HF_REPO, HF_FILE))
    if not path.is_file():
        raise SystemExit(f"model not found: {path}")
    torch.set_grad_enabled(False)
    model = ASRModel.restore_from(str(path), map_location=torch.device(device))
    model.encoder.set_default_att_context_size(context)  # also re-derives the streaming chunk sizes
    model.change_decoding_strategy(OmegaConf.structured(RNNTDecodingConfig(fused_batch_size=-1)), verbose=False)
    model.set_inference_prompt("fa-IR")
    model.decoding.set_strip_lang_tags(False, lang_tag_pattern=None)
    model = model.to(device=device, dtype=torch.float32).eval()
    return model


def text_of(hyps) -> str:
    h = hyps[0]
    return h.text if hasattr(h, "text") else h


def stream(model, audio: np.ndarray, realtime: bool, show_partial: bool) -> str:
    """Feed one recording through the model chunk by chunk; return the final transcript."""
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
    buf.append_audio(audio, stream_id=-1)
    cfg = model.encoder.streaming_cfg
    shift = cfg.shift_size[1] if isinstance(cfg.shift_size, list) else cfg.shift_size
    chunk_seconds = shift * model.cfg.preprocessor.window_stride  # feature frames -> seconds

    cache_ch, cache_t, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
    prev_hyps, prev_out, hyps, last = None, None, None, ""
    for step, (chunk, lengths) in enumerate(buf):
        t0 = time.perf_counter()
        with torch.inference_mode():
            prev_out, hyps, cache_ch, cache_t, cache_len, prev_hyps = model.conformer_stream_step(
                processed_signal=chunk.to(torch.float32), processed_signal_length=lengths,
                cache_last_channel=cache_ch, cache_last_time=cache_t, cache_last_channel_len=cache_len,
                keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=prev_hyps,
                previous_pred_out=prev_out,
                drop_extra_pre_encoded=0 if step == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded,
                return_transcription=True,
            )
        text = text_of(hyps)
        if show_partial and text != last:
            print(f"  … {text}", flush=True)
            last = text
        if realtime:
            time.sleep(max(0.0, chunk_seconds - (time.perf_counter() - t0)))
    return text_of(hyps) if hyps is not None else ""


@app.command()
def main(
    files: Annotated[list[Path] | None, typer.Argument(help="audio or video files")] = None,
    manifest: Annotated[Path | None, typer.Option(help="NeMo manifest; prints reference vs hypothesis + WER")] = None,
    record_seconds: Annotated[float | None, typer.Option("--record", help="record N seconds from the mic")] = None,
    mic: Annotated[str, typer.Option(help="avfoundation audio device index (see: ffmpeg -f avfoundation -list_devices true -i '')")] = "0",
    model_path: Annotated[Path | None, typer.Option("--model", help="local .nemo (default: download from Hugging Face)")] = None,
    lookahead: Annotated[str, typer.Option(help="look-ahead in seconds: 1.1 (more accurate) or 0.3 (lower latency)")] = "1.1",
    device: Annotated[str, typer.Option(help="auto, cpu, mps or cuda")] = "auto",
    realtime: Annotated[bool, typer.Option(help="pace chunks at real speaking speed")] = False,
    partial: Annotated[bool, typer.Option(help="print the growing transcript after each chunk")] = True,
) -> None:
    if lookahead not in CONTEXTS:
        raise SystemExit(f"--lookahead must be one of {list(CONTEXTS)}")
    items: list[dict] = []
    if manifest:
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                p = Path(r["audio_filepath"])
                r["path"] = p if p.is_absolute() else manifest.parent / p
                items.append(r)
    for f in files or []:
        items.append({"path": f})
    if record_seconds:
        items.append({"path": record(record_seconds, mic)})
    if not items:
        raise SystemExit("nothing to transcribe: give audio files, --manifest or --record N (see --help)")

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_path.name if model_path else HF_REPO} (look-ahead {lookahead} s, {device})...", flush=True)
    t = time.perf_counter()
    model = load_model(model_path, CONTEXTS[lookahead], device)
    print(f"Loaded in {time.perf_counter() - t:.0f} s.\n")

    refs, hyps = [], []
    for it in items:
        audio = load_audio(it["path"])
        dur = len(audio) / SAMPLE_RATE
        print(f"=== {it['path'].name}  ({dur:.1f} s)" + (f"  [{it['source']}]" if "source" in it else ""))
        t = time.perf_counter()
        hyp = stream(model, audio, realtime, partial)
        took = time.perf_counter() - t
        print(f"HYP: {hyp}")
        if "text" in it:
            from persian_asr.eval.metrics import score
            s = score([it["text"]], [hyp])
            print(f"REF: {it['text']}")
            print(f"WER {s.wer:.1%}  CER {s.cer:.1%}")
            refs.append(it["text"])
            hyps.append(hyp)
        print(f"({took:.1f} s to process {dur:.1f} s of audio)\n")

    if len(refs) > 1:
        from persian_asr.eval.metrics import score
        s = score(refs, hyps)
        print(f"Overall on {len(refs)} clips: WER {s.wer:.1%}  CER {s.cer:.1%}")


if __name__ == "__main__":
    app()
