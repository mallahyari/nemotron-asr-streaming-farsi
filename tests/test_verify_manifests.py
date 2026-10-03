import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_manifests.py"
REF = "ابپتثجچحخد"  # 10 letters; 2.0 s -> 5 chars/s


def build(tmp_path, mutate=None):
    (tmp_path / "manifests").mkdir(exist_ok=True)
    audio = tmp_path / "a.flac"
    sf.write(audio, np.zeros(32000, dtype=np.float32), 16000)
    rows = {
        "train": [{"id": "t1", "source": "yt", "speaker": "s1", "split": "train"}],
        "dev": [{"id": "d1", "source": "yt", "speaker": "h1", "split": "dev"}],
        "test": [{"id": "e1", "source": "yt", "speaker": "h2", "split": "test"}],
    }
    for split, rs in rows.items():
        for r in rs:
            r.update(text=REF, duration=2.0, audio_filepath=str(audio), target_lang="fa-IR")
    selected = [dict(r) for rs in rows.values() for r in rs]
    heldout = {"speakers": [["yt", "h1"], ["yt", "h2"]], "dev": ["d1"], "test": ["e1"]}
    scores = [{"id": r["id"], "hyp": REF} for r in selected]
    drops = []
    if mutate:
        mutate(rows, scores, drops, heldout)
    for split, rs in rows.items():
        (tmp_path / "manifests" / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
    (tmp_path / "selected.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scores))
    (tmp_path / "drops.jsonl").write_text("".join(json.dumps(d) + "\n" for d in drops))
    (tmp_path / "heldout.json").write_text(json.dumps(heldout))
    return subprocess.run([sys.executable, str(SCRIPT), "--work-dir", str(tmp_path), "--heldout", str(tmp_path / "heldout.json"),
                           "--drop-list", str(tmp_path / "drops.jsonl"), "--max-cer-source", "yt=0.6"],
                          capture_output=True, text=True)


def test_clean_set_passes(tmp_path):
    p = build(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "problems: NONE" in p.stdout


@pytest.mark.parametrize("name, mutate", [
    ("drop-listed clip present", lambda rows, s, d, h: d.append({"id": "t1", "reason": "language_id:en"})),
    ("train CER above cutoff", lambda rows, s, d, h: s.__setitem__(0, {"id": "t1", "hyp": "ابپسشصضطظع"})),  # 0.7 > 0.6
    ("held-out speaker in train", lambda rows, s, d, h: h["speakers"].append(["yt", "s1"])),
    ("text not canonical", lambda rows, s, d, h: rows["train"][0].update(text="كتاب")),  # Arabic kaf
    ("held-out id not in frozen list", lambda rows, s, d, h: h["dev"].clear()),
])
def test_each_violation_is_caught(tmp_path, name, mutate):
    p = build(tmp_path, mutate)
    assert p.returncode != 0 and name in p.stdout, p.stdout + p.stderr
