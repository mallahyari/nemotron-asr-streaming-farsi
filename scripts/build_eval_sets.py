"""Public Persian evaluation sets as 16 kHz NeMo manifests.

    uv run python scripts/build_eval_sets.py --out-dir /mnt/asr/eval

  fleurs_test   google/fleurs, fa_ir test         (CC-BY-4.0)  871 utterances, ~3.7 h
  cv_test       Common Voice 22.0 fa, test split   (CC0-1.0)    via fsicoli/common_voice_22_0

Both are pinned to a dataset commit, so a rerun gets identical files.

Reference text: `to_training_text` of the ORIGINAL transcription, the same
canonical form our model is trained on (FLEURS: the raw transcription column,
not FLEURS' own normalization). Utterances whose original text contains Latin
script are excluded: normalization would delete words that are actually
spoken, leaving a wrong reference. Exclusions are counted in the report.

FLEURS test.tsv has no header; its columns were identified from the data:
id (sentence id, repeated across speakers), file name, raw transcription,
normalized transcription, characters, num_samples (16 kHz), gender.
"""

import csv
import json
import subprocess
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import typer
from typing_extensions import Annotated

from persian_asr.text.asr_text import reject_reason, to_training_text

app = typer.Typer(pretty_exceptions_show_locals=False)

LANG = "fa-IR"
FLEURS = {"repo": "google/fleurs", "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
          "tsv": "data/fa_ir/test.tsv", "tar": "data/fa_ir/audio/test.tar.gz"}
CV = {"repo": "fsicoli/common_voice_22_0", "revision": "ae911de250fe9375d08e2daa2105b671d659d446",
      "tsv": "transcript/fa/test.tsv", "tar": "audio/fa/test/fa_test_0.tar"}
FLEURS_COLUMNS = ("id", "file", "raw", "normalized", "chars", "num_samples", "gender")


def download(spec: dict, cache: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    get = lambda f: Path(hf_hub_download(spec["repo"], f, repo_type="dataset", revision=spec["revision"], cache_dir=cache))
    return get(spec["tsv"]), get(spec["tar"])


def extract(tar_path: Path, wanted: set[str], dest: Path) -> dict[str, Path]:
    """Extract members whose basename is in `wanted`; returns basename -> path."""
    dest.mkdir(parents=True, exist_ok=True)
    found = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            name = Path(m.name).name
            if m.isfile() and name in wanted:
                if name in found:
                    raise SystemExit(f"{tar_path.name}: duplicate member {name}")
                f = tf.extractfile(m)
                target = dest / name
                target.write_bytes(f.read())
                found[name] = target
    return found


def to_flac(job: tuple[str, str]) -> tuple[str, float | None, str]:
    """Any audio file -> 16 kHz mono 16-bit FLAC; returns (dst, duration, error)."""
    import soundfile as sf

    src, dst = job
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000",
                        "-sample_fmt", "s16", dst], capture_output=True, text=True)
    if p.returncode != 0:
        return dst, None, p.stderr.strip()[-300:]
    info = sf.info(dst)
    return dst, info.frames / info.samplerate, ""


def build(name: str, items: list[dict], raw_dir: Path, out_dir: Path, workers: int) -> dict:
    """items: {id, file, raw_text, speaker, extra...} -> manifest + stats."""
    audio_dir = out_dir / name
    audio_dir.mkdir(parents=True, exist_ok=True)
    excluded = Counter()
    keep = []
    for it in items:
        reason = reject_reason(it["raw_text"])
        if reason == "latin_script" or not it["raw_text"].strip():
            excluded[reason or "empty"] += 1
            continue
        keep.append(it)
    jobs = [(str(raw_dir / it["file"]), str(audio_dir / f"{it['id']}.flac")) for it in keep]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(to_flac, jobs, chunksize=16))
    rows = []
    for it, (dst, dur, err) in zip(keep, results):
        if dur is None or dur < 0.2:
            excluded["audio_error"] += 1
            print(f"  {it['id']}: {err or f'duration {dur}'}")
            continue
        rows.append({"audio_filepath": dst, "duration": round(dur, 3), "text": to_training_text(it["raw_text"]),
                     "target_lang": LANG, "lang": LANG, "id": it["id"], "source": name, "speaker": it["speaker"]})
    with open(out_dir / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"utterances": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
             "speakers": len({r["speaker"] for r in rows}), "excluded": dict(excluded)}
    print(f"{name}: {stats}")
    return stats


@app.command()
def main(
    out_dir: Annotated[Path, typer.Option(help="new directory for the eval manifests and audio")] = Path("/mnt/asr/eval"),
    workers: int = 16,
) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty")
    cache = out_dir / "_downloads"
    report = {}

    tsv, tar = download(FLEURS, cache)
    with open(tsv, newline="") as f:
        rows = [dict(zip(FLEURS_COLUMNS, r)) for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    if any(len(r) != len(FLEURS_COLUMNS) for r in rows):
        raise SystemExit("FLEURS test.tsv: unexpected column count")
    files = extract(tar, {r["file"] for r in rows}, cache / "fleurs_wav")
    if missing := {r["file"] for r in rows} - set(files):
        raise SystemExit(f"FLEURS: {len(missing)} audio files missing from the archive")
    items = [{"id": f"fleurs_{Path(r['file']).stem}", "file": r["file"], "raw_text": r["raw"],
              # FLEURS has no speaker ids; each recording is treated as its own speaker
              "speaker": f"fleurs_{Path(r['file']).stem}"} for r in rows]
    report["fleurs_test"] = build("fleurs_test", items, cache / "fleurs_wav", out_dir, workers)
    report["fleurs_test"]["source"] = {k: FLEURS[k] for k in ("repo", "revision")}

    tsv, tar = download(CV, cache)
    with open(tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    files = extract(tar, {r["path"] for r in rows}, cache / "cv_mp3")
    if missing := {r["path"] for r in rows} - set(files):
        raise SystemExit(f"Common Voice: {len(missing)} audio files missing from the archive")
    items = [{"id": f"cv_{Path(r['path']).stem}", "file": r["path"], "raw_text": r["sentence"],
              "speaker": f"cv_{r['client_id'][:16]}"} for r in rows]
    report["cv_test"] = build("cv_test", items, cache / "cv_mp3", out_dir, workers)
    report["cv_test"]["source"] = {k: CV[k] for k in ("repo", "revision")}

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()
