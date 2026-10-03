"""Independent verification of the final train/dev/test manifests.

    uv run python scripts/verify_manifests.py --work-dir /mnt/asr \\
        --heldout data/splits/heldout_v1.json --drop-list /mnt/asr/lid_drops.jsonl \\
        --max-cer 0.8 --max-cer-source farsi_asr_yt=0.6 --max-cer-source youtube=0.6

Re-derives every property from the pipeline's inputs instead of trusting its
outputs, and exits non-zero if any row violates one:

  ids unique; split, text and duration agree with selected.jsonl; dev/test ids
  in the frozen held-out list; text canonical (to_training_text(t) == t), only
  Persian letters + space + ZWNJ, no stretched letters; duration and speaking
  rate inside the select limits; target_lang fa-IR; audio exists; a sample of
  audio headers is 16 kHz mono with the manifest's duration; train CER (recomputed
  from scores.jsonl) within the per-source cutoff; no drop-listed clip anywhere;
  no held-out speaker in train.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_show_locals=False)

SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@app.command()
def main(
    work_dir: Annotated[Path, typer.Option()],
    heldout: Annotated[Path, typer.Option(help="frozen split file")],
    drop_list: Annotated[list[Path] | None, typer.Option()] = None,
    max_cer: float = 0.8,
    max_cer_source: Annotated[list[str] | None, typer.Option(help="SOURCE=VALUE")] = None,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    min_cps: float = 1.5,
    max_cps: float = 25.0,
    audio_sample: int = 1500,
) -> None:
    import jiwer
    import soundfile as sf

    from persian_asr.text.asr_text import to_scoring_text, to_training_text
    from persian_asr.text.normalize_fa import PERSIAN_LETTERS, ZWNJ

    cutoff = defaultdict(lambda: max_cer)
    for item in max_cer_source or []:
        src, _, v = item.partition("=")
        cutoff[src] = float(v)
    dropped = {d["id"] for p in drop_list or [] for d in read_jsonl(p)}
    sel = {r["id"]: r for r in read_jsonl(work_dir / "selected.jsonl")}
    hyp = {r["id"]: r["hyp"] for r in read_jsonl(work_dir / "scores.jsonl")}
    frozen = json.loads(heldout.read_text())
    held_ids = {"dev": set(frozen["dev"]), "test": set(frozen["test"])}
    held_spk = {tuple(s) for s in frozen["speakers"]}
    allowed = set(PERSIAN_LETTERS) | {" ", ZWNJ}
    stretched = re.compile(rf"([{PERSIAN_LETTERS}])\1\1")

    problems: Counter = Counter()
    seen: set = set()
    totals = {}
    for split in SPLITS:
        rows = read_jsonl(work_dir / "manifests" / f"{split}.jsonl")
        totals[split] = (len(rows), round(sum(r["duration"] for r in rows) / 3600, 2))
        for r in rows:
            s = sel.get(r["id"])
            checks = {
                "duplicate id": r["id"] in seen,
                "not in selected.jsonl": s is None,
                "split != selected": s is not None and s["split"] != split,
                "held-out id not in frozen list": split != "train" and r["id"] not in held_ids[split],
                "held-out speaker in train": split == "train" and (r["source"], r["speaker"]) in held_spk,
                "text != selected": s is not None and r["text"] != s["text"],
                "empty text / bad characters": not r["text"] or bool(set(r["text"]) - allowed),
                "text not canonical": to_training_text(r["text"]) != r["text"],
                "stretched letters": bool(stretched.search(r["text"])),
                "duration out of range": not min_duration - 0.1 <= r["duration"] <= max_duration + 0.1,
                "duration != selected": s is not None and abs(r["duration"] - s["duration"]) > 0.1,
                "speaking rate out of range": s is not None
                and not min_cps <= len(r["text"].replace(" ", "")) / s["duration"] <= max_cps,
                "target_lang": r.get("target_lang") != "fa-IR",
                "audio missing": not Path(r["audio_filepath"]).exists(),
                "drop-listed clip present": r["id"] in dropped,
                "no score": r["id"] not in hyp,
            }
            if split == "train" and r["id"] in hyp:
                cer = jiwer.cer(to_scoring_text(r["text"]), to_scoring_text(hyp[r["id"]]))
                checks["train CER above cutoff"] = cer > cutoff[r["source"]]
            seen.add(r["id"])
            for name, bad in checks.items():
                problems[name] += bool(bad)
        for r in random.Random(0).sample(rows, min(audio_sample, len(rows))):
            info = sf.info(r["audio_filepath"])
            bad = (info.samplerate, info.channels) != (16000, 1) or abs(info.frames / info.samplerate - r["duration"]) > 0.002
            problems["audio header (sampled)"] += bad
    found = {k: v for k, v in problems.items() if v}
    print("totals (clips, hours):", totals)
    print("problems:", found or "NONE")
    if found:
        raise SystemExit(1)


if __name__ == "__main__":
    app()
