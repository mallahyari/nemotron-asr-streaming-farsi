import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_asr_data.py"

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def run(*args):
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True, capture_output=True, text=True)


def test_select_and_cut(tmp_path):
    # One 10 s, 44.1 kHz stereo "recording" that rows window into, like farsi_asr_yt.
    rec = tmp_path / "rec.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
                    "-ac", "2", "-ar", "44100", str(rec)], check=True)
    rows = [
        {"path": str(rec), "start": 1.0, "duration": 2.5, "transcript": "او می رود.", "speaker": "a", "source": "yt"},
        {"path": str(rec), "start": 4.0, "duration": 3.0, "transcript": "نمیدونم میشه یا نه", "speaker": "b", "source": "yt"},
        # dropped: Latin script, too short, duplicate window
        {"path": str(rec), "start": 0.0, "duration": 2.0, "transcript": "hello دنیا", "speaker": "a", "source": "yt"},
        {"path": str(rec), "start": 8.0, "duration": 0.2, "transcript": "سلام", "speaker": "a", "source": "yt"},
        {"path": str(rec), "start": 1.0, "duration": 2.5, "transcript": "او می رود.", "speaker": "a", "source": "yt"},
        # windows past the end of the 10 s recording: truncated, and starting after the end
        {"path": str(rec), "start": 8.0, "duration": 4.0, "transcript": "این قطع شده", "speaker": "c", "source": "yt"},
        {"path": str(rec), "start": 12.0, "duration": 3.0, "transcript": "این وجود ندارد", "speaker": "c", "source": "yt"},
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    work = tmp_path / "work"

    run("select", "--raw", str(raw), "--work-dir", str(work))
    selected = [json.loads(l) for l in (work / "selected.jsonl").read_text().splitlines()]
    assert [r["text"] for r in selected] == ["او می‌رود", "نمی‌دونم می‌شه یا نه", "این قطع شده", "این وجود ندارد"]
    assert {r["split"] for r in selected} == {"train"}  # < 20 speakers: nothing held out

    run("cut", "--work-dir", str(work), "--workers", "2")
    cut = [json.loads(l) for l in (work / "cut_train.jsonl").read_text().splitlines()]
    assert len(cut) == 2
    for r in cut:
        info = sf.info(r["audio_filepath"])
        assert (info.samplerate, info.channels) == (16000, 1)
        assert r["target_lang"] == "fa-IR"
    assert sorted(r["duration"] for r in cut) == pytest.approx([2.5, 3.0], abs=0.05)
    errors = [json.loads(l) for l in (work / "cut_errors.jsonl").read_text().splitlines()]
    assert len(errors) == 2  # the truncated window and the one starting after the end


def test_assign_splits_many_speakers_capped():
    import importlib.util
    import random

    spec = importlib.util.spec_from_file_location("prep", SCRIPT)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    # 100 speakers, 10..109 clips of 30 s each (5..54.5 min).
    rows = [{"source": "yt", "speaker": f"s{i}", "duration": 30.0} for i in range(100) for _ in range(10 + i)]
    prep.assign_splits(rows, dev_speakers=5, test_speakers=8, minutes=6.0, rng=random.Random(0))
    by_split: dict = {}
    for r in rows:
        by_split.setdefault(r["split"], {}).setdefault(r["speaker"], []).append(r)
    assert len(by_split["test"]) == 8 and len(by_split["dev"]) == 5
    # each held-out speaker: at most 6 min (+ the clip that crossed the budget)
    assert all(sum(r["duration"] for r in v) <= 6.5 * 60 for v in by_split["test"].values())
    # held-out speakers never appear in train
    assert not (set(by_split["train"]) & (set(by_split["test"]) | set(by_split["dev"])))
    # the remainder of held-out speakers is excluded, not trained on
    assert set(by_split["excluded"]) <= set(by_split["test"]) | set(by_split["dev"])


REF = "ابپتثجچحخد"


def test_finalize_filters_train_only(tmp_path):
    def manifest(split, rows):
        (tmp_path / f"cut_{split}.jsonl").write_text(
            "".join(json.dumps({"id": i, "source": "yt", "duration": 3600.0, "text": REF}, ensure_ascii=False) + "\n" for i in rows)
        )
    manifest("train", ["t_good", "t_edge", "t_bad", "t_insert"])
    manifest("dev", ["d_bad"])
    manifest("test", ["e_good"])
    # The stored "cer" is deliberately wrong (0.0): finalize must recompute it from hyp vs text.
    # REF is 10 distinct Persian letters, so CER = character edits / 10 (no letter repeats 3x,
    # so the stretched-letter rule can't change anything).
    hyps = {
        "t_good": "ابپتثجچحخذ",            # 1 substitution   -> 0.1
        "t_edge": "ابسشصضطظعغ",            # 8 substitutions  -> 0.8 (kept: <= cutoff)
        "t_bad": "اسشصضطظعغف",             # 9 substitutions  -> 0.9
        "t_insert": REF + "سشصضطظعغفقلم",  # 12 insertions    -> 1.2
        "d_bad": "",                       # empty            -> 1.0 (dev: kept anyway)
        "e_good": REF,                     # exact            -> 0.0
    }
    (tmp_path / "scores.jsonl").write_text(
        "".join(json.dumps({"id": i, "hyp": h, "cer": 0.0}, ensure_ascii=False) + "\n" for i, h in hyps.items())
    )

    run("finalize", "--work-dir", str(tmp_path), "--max-cer", "0.8")
    ids = lambda s: [json.loads(l)["id"] for l in (tmp_path / "manifests" / f"{s}.jsonl").read_text().splitlines()]
    assert ids("train") == ["t_good", "t_edge"]  # <= cutoff kept, > cutoff (incl. CER > 1) dropped
    assert ids("dev") == ["d_bad"] and ids("test") == ["e_good"]  # held-out sets never filtered
    report = json.loads((tmp_path / "manifests" / "report.json").read_text())
    assert report["train"]["hours"] == 2.0 and report["train"]["dropped_hours"] == 2.0
    assert sum(report["train"]["cer_histogram_hours"]["yt"].values()) == 4.0


def test_finalize_refuses_missing_scores(tmp_path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"cut_{split}.jsonl").write_text(json.dumps({"id": split, "source": "yt", "duration": 1.0, "text": "x"}) + "\n")
    (tmp_path / "scores.jsonl").write_text(json.dumps({"id": "train", "hyp": "", "cer": 0.0}) + "\n")
    with pytest.raises(subprocess.CalledProcessError):
        run("finalize", "--work-dir", str(tmp_path))
    assert not (tmp_path / "manifests").exists()  # nothing written when any split lacks scores


def _prep_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("prep", SCRIPT)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    return prep


def test_frozen_heldout_roundtrip_and_stability():
    import random

    prep = _prep_module()
    rows = [{"id": f"s{i}_{j}", "source": "yt", "speaker": f"s{i}", "duration": 30.0}
            for i in range(100) for j in range(10 + i)]
    prep.assign_splits(rows, dev_speakers=5, test_speakers=8, minutes=6.0, rng=random.Random(0))
    original = {r["id"]: r["split"] for r in rows}
    frozen = json.loads(json.dumps(prep.heldout_of(rows)))  # as it round-trips through heldout.json

    # 1. reapplying the frozen split reproduces it exactly
    again = [dict(r) for r in rows]
    prep.apply_heldout(again, frozen)
    assert {r["id"]: r["split"] for r in again} == original

    # 2. a filter change (drop some clips, add a new clip for a held-out and a train speaker)
    held_spk = next(r["speaker"] for r in rows if r["split"] == "test")
    train_spk = next(r["speaker"] for r in rows if r["split"] == "train")
    changed = [dict(r) for r in rows[::2]] + [
        {"id": "new_held", "source": "yt", "speaker": held_spk, "duration": 30.0},
        {"id": "new_train", "source": "yt", "speaker": train_spk, "duration": 30.0},
    ]
    prep.apply_heldout(changed, frozen)
    split = {r["id"]: r["split"] for r in changed}
    assert split["new_held"] == "excluded" and split["new_train"] == "train"
    for r in changed[:-2]:
        assert r["split"] == original[r["id"]]  # surviving clips never move
    spk = lambda s: {r["speaker"] for r in changed if r["split"] == s}
    assert not spk("train") & (spk("dev") | spk("test") | spk("excluded"))


def test_finalize_per_source_cutoff_and_drop_lists(tmp_path):
    def manifest(split, rows):
        (tmp_path / f"cut_{split}.jsonl").write_text("".join(
            json.dumps({"id": i, "source": src, "duration": 3600.0, "text": REF}, ensure_ascii=False) + "\n"
            for i, src in rows))
    # 7 substitutions -> CER 0.7 for every clip
    hyp07 = "ابپسشصضطظع"
    manifest("train", [("yt_a", "youtube"), ("fi_a", "filimo"), ("fi_lid", "filimo")])
    manifest("dev", [("dev_lid", "youtube"), ("dev_ok", "youtube")])
    manifest("test", [("test_ok", "filimo")])
    ids = ["yt_a", "fi_a", "fi_lid", "dev_lid", "dev_ok", "test_ok"]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps({"id": i, "hyp": hyp07, "cer": 0.0}, ensure_ascii=False) + "\n" for i in ids))
    (tmp_path / "drops.jsonl").write_text("".join(json.dumps({"id": i, "reason": "language_id:en"}) + "\n" for i in ("fi_lid", "dev_lid", "not_in_any_manifest")))

    run("finalize", "--work-dir", str(tmp_path), "--max-cer", "0.8", "--max-cer-source", "youtube=0.6",
        "--drop-list", str(tmp_path / "drops.jsonl"))
    ids_of = lambda s: [json.loads(l)["id"] for l in (tmp_path / "manifests" / f"{s}.jsonl").read_text().splitlines()]
    assert ids_of("train") == ["fi_a"]        # youtube 0.7 > 0.6 dropped; filimo 0.7 <= 0.8 kept; fi_lid listed
    assert ids_of("dev") == ["dev_ok"]        # drop list applies to dev too (CER cutoff does not)
    assert ids_of("test") == ["test_ok"]
    rep = json.loads((tmp_path / "manifests" / "report.json").read_text())
    assert rep["train"]["max_cer"] == {"filimo": 0.8, "youtube": 0.6}
    assert rep["train"]["dropped_by_list"] == {"language_id:en": {"clips": 1, "hours": 1.0}}
    assert rep["dev"]["dropped_by_list"] == {"language_id:en": {"clips": 1, "hours": 1.0}}


def test_finalize_rejects_malformed_source_cutoff(tmp_path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"cut_{split}.jsonl").write_text("")
    (tmp_path / "scores.jsonl").write_text("")
    with pytest.raises(subprocess.CalledProcessError):
        run("finalize", "--work-dir", str(tmp_path), "--max-cer-source", "youtube0.6")
