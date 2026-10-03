"""Turn the raw Persian speech manifests into NeMo ASR training data.

Four stages, each resumable and each writing into --work-dir:

    select   raw manifests -> selected.jsonl
             normalize text (to_training_text), drop rows by text/duration/
             speaking-rate rules, assign speaker-disjoint dev/test splits.
    cut      selected.jsonl -> audio/**.flac + cut_<split>.jsonl
             cut each clip out of its source file, 16 kHz mono 16-bit FLAC.
    score    cut_*.jsonl -> scores.jsonl                         (GPU)
             transcribe every clip with an existing Persian ASR model and
             measure CER against the transcript: does the label match the audio?
    finalize cut_*.jsonl + scores.jsonl -> manifests/<split>.jsonl + report
             drop clips whose transcript disagrees with the audio.

    uv run python scripts/prepare_asr_data.py select --raw /mnt/data/farsi_600h/raw_manatts.jsonl ...
    uv run python scripts/prepare_asr_data.py cut
    uv run python scripts/prepare_asr_data.py score
    uv run python scripts/prepare_asr_data.py finalize

Input rows (pocket-tts manifest format): path, start, duration, transcript,
speaker, source. `start`/`duration` window into `path`, which may be a whole
recording (farsi_asr_yt) or an already-cut clip (start 0).

See REPRODUCE.md for the full walkthrough.
"""

import hashlib
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import typer
from typing_extensions import Annotated

from persian_asr.text.asr_text import reject_reason, to_scoring_text, to_training_text

app = typer.Typer(pretty_exceptions_show_locals=False)

SAMPLE_RATE = 16000
LANG = "fa-IR"  # Nemotron 3.5 prompt-dictionary slot 38
SPLITS = ("train", "dev", "test")
# A cut must match its requested window to within this (mp3 frames are ~26 ms).
CUT_TOLERANCE = 0.1

WorkDir = Annotated[Path, typer.Option(help="where every stage reads and writes")]
DEFAULT_WORK = Path("/mnt/asr")


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def hours(rows) -> float:
    return sum(r["duration"] for r in rows) / 3600


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def text_reject(raw_text: str, text: str, duration: float, limits: dict) -> str | None:
    if (reason := reject_reason(raw_text)) is not None:
        return reason
    if not limits["min_duration"] <= duration <= limits["max_duration"]:
        return "duration"
    # Characters per second, spaces excluded. Far too few means the transcript
    # covers only part of the audio; far too many means the clip was cut short.
    cps = len(text.replace(" ", "")) / duration
    if not limits["min_cps"] <= cps <= limits["max_cps"]:
        return "speaking_rate"
    return None


def assign_splits(rows: list[dict], dev_speakers: int, test_speakers: int, minutes: float, rng: random.Random) -> None:
    """Set r["split"]: hold out whole speakers (videos/films) per source for dev/test.

    Held-out sets should cover MANY unseen voices, so each held-out speaker
    contributes at most `minutes` of audio; the rest of that speaker's clips
    become "excluded" (in neither train nor eval -- putting them in train
    would leak the speaker). Candidates are the speakers with no more audio
    than the source's median, which keeps the excluded remainder small and
    the big speakers in train.
    """
    by_spk: dict = defaultdict(list)
    for r in rows:
        r["split"] = "train"
        by_spk[(r["source"], r["speaker"])].append(r)
    by_source: dict = defaultdict(list)
    for (source, spk) in by_spk:
        by_source[source].append(spk)
    for source, spks in sorted(by_source.items()):
        if len(spks) < 20:
            continue  # single narrator (Mana-TTS): nothing speaker-disjoint to hold out
        h = {s: hours(by_spk[(source, s)]) for s in spks}
        median = sorted(h.values())[len(h) // 2]
        candidates = sorted(s for s in spks if h[s] <= median)
        rng.shuffle(candidates)
        for split, n in (("test", test_speakers), ("dev", dev_speakers)):
            for s in candidates[:n]:
                clips = by_spk[(source, s)]
                rng.shuffle(clips)
                budget = minutes * 60
                for r in clips:
                    r["split"] = split if budget > 0 else "excluded"
                    budget -= r["duration"]
            candidates = candidates[n:]


def apply_heldout(rows: list[dict], heldout: dict) -> None:
    """Set r["split"] from a frozen held-out file instead of drawing speakers.

    `heldout` = {"speakers": [[source, speaker], ...], "dev": [ids], "test": [ids]}.
    Clips listed as dev/test keep that split; any other clip of a held-out
    speaker is excluded (never train -- that would leak the speaker); every
    remaining clip is train. Filters may change between runs; the held-out
    speakers never do.
    """
    held_spk = {tuple(s) for s in heldout["speakers"]}
    split_of = {i: "dev" for i in heldout["dev"]} | {i: "test" for i in heldout["test"]}
    for r in rows:
        if r["id"] in split_of:
            r["split"] = split_of[r["id"]]
        elif (r["source"], r["speaker"]) in held_spk:
            r["split"] = "excluded"
        else:
            r["split"] = "train"


def heldout_of(rows: list[dict]) -> dict:
    """The frozen-split record for `rows` (inverse of apply_heldout)."""
    held = [r for r in rows if r["split"] != "train"]
    return {
        "speakers": sorted({(r["source"], r["speaker"]) for r in held}),
        "dev": sorted(r["id"] for r in held if r["split"] == "dev"),
        "test": sorted(r["id"] for r in held if r["split"] == "test"),
    }


@app.command()
def select(
    raw: Annotated[list[Path], typer.Option(help="raw pocket-tts manifests (repeatable)")],
    work_dir: WorkDir = DEFAULT_WORK,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    min_cps: float = 1.5,
    max_cps: float = 25.0,
    dev_speakers: Annotated[int, typer.Option(help="held-out speakers per conversational source")] = 20,
    test_speakers: Annotated[int, typer.Option(help="held-out speakers per conversational source")] = 30,
    heldout_minutes: Annotated[float, typer.Option(help="max audio per held-out speaker")] = 6.0,
    seed: int = 0,
    heldout: Annotated[
        Path | None, typer.Option(help="frozen split (heldout.json from an earlier run); reused instead of drawing speakers")
    ] = None,
) -> None:
    """Normalize, filter and split the raw manifests.

    Always writes <work-dir>/heldout.json, the split it used. Pass it back
    with --heldout on later runs so changing a filter never changes dev/test.
    """
    limits = dict(min_duration=min_duration, max_duration=max_duration, min_cps=min_cps, max_cps=max_cps)
    kept, reasons, seen = [], Counter(), set()
    for path in raw:
        for r in read_jsonl(path):
            text = to_training_text(r["transcript"])
            if (reason := text_reject(r["transcript"], text, r["duration"], limits)) is not None:
                reasons[(r["source"], reason)] += 1
                continue
            key = (r["path"], round(r.get("start", 0.0), 2))
            if key in seen:
                reasons[(r["source"], "duplicate")] += 1
                continue
            seen.add(key)
            uid = r["source"] + "_" + hashlib.sha1(f"{key[0]}|{key[1]}".encode()).hexdigest()[:14]
            kept.append({
                "id": uid, "src_path": r["path"], "start": r.get("start", 0.0),
                "duration": r["duration"], "text": text,
                "speaker": r["speaker"], "source": r["source"],
            })
    if heldout is not None:
        apply_heldout(kept, json.loads(heldout.read_text()))
    else:
        assign_splits(kept, dev_speakers, test_speakers, heldout_minutes, random.Random(seed))
    n = write_jsonl(work_dir / "selected.jsonl", kept)
    (work_dir / "heldout.json").write_text(json.dumps(heldout_of(kept), ensure_ascii=False, indent=0) + "\n")

    print(f"kept {n:,} rows, {hours(kept):,.1f} h")
    by = defaultdict(list)
    for r in kept:
        by[(r["source"], r["split"])].append(r)
    for (source, split), rs in sorted(by.items()):
        print(f"  {source:<14} {split:<5} {len(rs):>8,} rows {hours(rs):8.1f} h  {len({r['speaker'] for r in rs}):>5} speakers")
    print("dropped:")
    for (source, reason), c in sorted(reasons.items()):
        print(f"  {source:<14} {reason:<18} {c:>8,}")


# ---------------------------------------------------------------------------
# cut
# ---------------------------------------------------------------------------


def audio_path(work_dir: Path, r: dict) -> Path:
    return work_dir / "audio" / r["source"] / r["id"][-2:] / f"{r['id']}.flac"


def cut_one(r: dict, dst: str) -> tuple[str, float | None, str]:
    """Cut [start, start+duration) out of src_path as 16 kHz mono FLAC."""
    if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part.flac"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{r['start']:.3f}", "-t", f"{r['duration']:.3f}",
               "-i", r["src_path"], "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", tmp]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return r["id"], None, proc.stderr.strip()[-300:]
        os.replace(tmp, dst)
    import soundfile as sf

    info = sf.info(dst)
    return r["id"], info.frames / info.samplerate, ""


@app.command()
def cut(work_dir: WorkDir = DEFAULT_WORK, workers: int = os.cpu_count() or 8) -> None:
    """Cut and resample every selected clip (CPU-bound; uses all cores)."""
    rows = {r["id"]: r for r in read_jsonl(work_dir / "selected.jsonl") if r["split"] in SPLITS}
    out: dict = {s: [] for s in SPLITS}
    errors = []
    from tqdm import tqdm

    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(cut_one, r, str(audio_path(work_dir, r))) for r in rows.values()]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="clip"):
            uid, dur, err = fut.result()
            r = rows[uid]
            # A cut that doesn't match its window means the window ran past the
            # end of the recording: the transcript covers audio that isn't there.
            # A window starting AFTER the end gives a header-only FLAC whose
            # length reads as garbage (~5.8e14 h), so check both directions.
            if dur is None or abs(dur - r["duration"]) > CUT_TOLERANCE:
                errors.append({"id": uid, "src_path": r["src_path"],
                               "error": err or f"cut {dur:.3f}s, window {r['duration']:.3f}s"})
                continue
            out[r["split"]].append({
                "audio_filepath": str(audio_path(work_dir, r)), "duration": round(dur, 3),
                "text": r["text"], "target_lang": LANG, "lang": LANG,
                "id": uid, "source": r["source"], "speaker": r["speaker"],
            })
    for split, rs in out.items():
        rs.sort(key=lambda r: r["id"])
        write_jsonl(work_dir / f"cut_{split}.jsonl", rs)
        print(f"{split:<5} {len(rs):>8,} clips {hours(rs):8.1f} h")
    write_jsonl(work_dir / "cut_errors.jsonl", errors)
    print(f"{len(errors):,} clips failed (cut_errors.jsonl)")


# ---------------------------------------------------------------------------
# score (GPU)
# ---------------------------------------------------------------------------


@app.command()
def score(
    work_dir: WorkDir = DEFAULT_WORK,
    model: str = "nvidia/stt_fa_fastconformer_hybrid_large",
    batch_size: int = 128,
    chunk: Annotated[int, typer.Option(help="clips per transcribe() call; progress is saved after each")] = 20000,
    num_workers: int = 8,
) -> None:
    """Transcribe every clip with an existing Persian model; record per-clip CER."""
    import jiwer
    import torch
    from nemo.collections.asr.models import ASRModel

    out_path = work_dir / "scores.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    rows = [r for s in SPLITS for r in read_jsonl(work_dir / f"cut_{s}.jsonl") if r["id"] not in done]
    print(f"{len(done):,} already scored, {len(rows):,} to go")
    # NeMo's loader allocates by duration: one garbage duration crashes the run.
    if bad := [r["id"] for r in rows if not 0 < r["duration"] <= 40]:
        raise SystemExit(f"{len(bad)} clips have implausible durations (e.g. {bad[:3]}); re-run `cut`")
    asr = ASRModel.from_pretrained(model).eval().to("cuda")
    if hasattr(asr, "ctc_decoder"):
        # Hybrid RNNT/CTC model: use the CTC head, the stronger one for
        # stt_fa (13.2 vs 15.5 WER), with batched greedy decoding.
        decoding = asr.cfg.aux_ctc.decoding
        decoding.strategy = "greedy_batch"
        asr.change_decoding_strategy(decoding_cfg=decoding, decoder_type="ctc")
        assert asr.cur_decoder == "ctc", asr.cur_decoder
    with open(out_path, "a") as f:
        for i in range(0, len(rows), chunk):
            # Longest first keeps each batch padded evenly.
            part = sorted(rows[i : i + chunk], key=lambda r: -r["duration"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                hyps = asr.transcribe(
                    [r["audio_filepath"] for r in part], batch_size=batch_size, num_workers=num_workers, verbose=False
                )
            if len(hyps) != len(part):  # e.g. a (best, all) tuple from an RNNT decoder
                raise RuntimeError(f"transcribe returned {len(hyps)} results for {len(part)} files")
            for r, h in zip(part, hyps):
                h = h.text if hasattr(h, "text") else str(h)
                ref, hyp = to_scoring_text(r["text"]), to_scoring_text(h)
                cer = jiwer.cer(ref, hyp) if ref else 1.0
                f.write(json.dumps({"id": r["id"], "hyp": h, "cer": round(cer, 4)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"scored {min(i + chunk, len(rows)):,}/{len(rows):,}")


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


@app.command()
def finalize(
    work_dir: WorkDir = DEFAULT_WORK,
    max_cer: Annotated[float, typer.Option(help="drop train clips whose label disagrees with the audio beyond this")] = 0.8,
    max_cer_source: Annotated[
        list[str] | None, typer.Option(help="per-source override for train, SOURCE=VALUE (repeatable)")
    ] = None,
    drop_list: Annotated[
        list[Path] | None,
        typer.Option(help="jsonl of {id, reason} to drop from ALL splits, e.g. lid_drops.jsonl (repeatable)"),
    ] = None,
) -> None:
    """Drop mislabelled clips; write final manifests and a report.

    - The CER cutoff applies to TRAIN only: filtering dev/test by the scoring
      model would bias evaluation towards what that model already gets right.
    - Drop lists apply to EVERY split: they mark label errors found
      independently of the scorer (e.g. non-Persian audio), which are wrong
      references in dev/test too.

    CER is recomputed here from the stored hypothesis and the CURRENT
    transcript, so a later text-normalization change can't leave stale scores.
    """
    import jiwer

    cutoff = defaultdict(lambda: max_cer)
    for item in max_cer_source or []:
        source, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--max-cer-source expects SOURCE=VALUE, got {item!r}")
        cutoff[source] = float(value)
    dropped_by: dict = {}
    for path in drop_list or []:
        for d in read_jsonl(path):
            dropped_by.setdefault(d["id"], d["reason"])

    hyps = {r["id"]: r["hyp"] for r in read_jsonl(work_dir / "scores.jsonl")}
    bins = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, float("inf"))
    cut_rows = {split: list(read_jsonl(work_dir / f"cut_{split}.jsonl")) for split in SPLITS}
    # Check every split before writing anything, so a failure leaves no partial output.
    for split, rows in cut_rows.items():
        if missing := [r for r in rows if r["id"] not in hyps]:
            raise SystemExit(f"{len(missing):,} {split} clips have no score; run `score` first")
    scores = {}
    for rows in cut_rows.values():
        for r in rows:
            ref, hyp = to_scoring_text(r["text"]), to_scoring_text(hyps[r["id"]])
            scores[r["id"]] = {"cer": jiwer.cer(ref, hyp) if ref else 1.0}
    report = {}
    for split, rows in cut_rows.items():
        hist: dict = defaultdict(Counter)
        for r in rows:
            c = scores[r["id"]]["cer"]
            hist[r["source"]][next(b for b in bins if c <= b)] += r["duration"] / 3600
        # dev/test are not filtered by the scoring model: that would bias the
        # evaluation towards what that model already gets right. They get
        # hand-checked instead.
        listed = [r for r in rows if r["id"] in dropped_by]
        kept = [r for r in rows if r["id"] not in dropped_by]
        if split == "train":
            kept = [r for r in kept if scores[r["id"]]["cer"] <= cutoff[r["source"]]]
        write_jsonl(work_dir / "manifests" / f"{split}.jsonl", kept)
        reasons = Counter(dropped_by[r["id"]] for r in listed)
        report[split] = {
            "max_cer": ({s: cutoff[s] for s in sorted({r["source"] for r in rows})} if split == "train" else None),
            "clips": len(kept), "hours": round(hours(kept), 2),
            "dropped_hours": round(hours(rows) - hours(kept), 2),
            "dropped_by_list": {k: {"clips": v, "hours": round(hours([r for r in listed if dropped_by[r["id"]] == k]), 2)}
                                for k, v in sorted(reasons.items())},
            "by_source_hours": {s: round(hours([r for r in kept if r["source"] == s]), 2) for s in sorted({r["source"] for r in rows})},
            "cer_histogram_hours": {s: {f"<={b}": round(h[b], 2) for b in bins} for s, h in hist.items()},
        }
        print(f"{split:<5} kept {len(kept):>8,} clips {hours(kept):8.1f} h  (dropped {report[split]['dropped_hours']:.1f} h)")
    (work_dir / "manifests" / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {work_dir / 'manifests'}")


if __name__ == "__main__":
    app()
