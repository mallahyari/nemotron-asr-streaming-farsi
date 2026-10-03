import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf

from persian_asr.text.asr_text import to_training_text

# Import under its real module name via sys.path: build() uses a process pool,
# and spawned workers (macOS) must be able to re-import the module by name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
bes = importlib.import_module("build_eval_sets")

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_build_converts_normalizes_and_excludes_latin(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # 48 kHz stereo mp3, like Common Voice
    for name in ("a.mp3", "b.mp3"):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
                        "-ac", "2", "-ar", "48000", str(raw / name)], check=True)
    items = [
        {"id": "cv_a", "file": "a.mp3", "raw_text": "سال ۱۴۰۲ آمریكا، می رود.", "speaker": "s1"},
        {"id": "cv_b", "file": "b.mp3", "raw_text": "این NASA است", "speaker": "s2"},  # Latin -> excluded
    ]
    stats = bes.build("cv_test", items, raw, tmp_path / "out", workers=2)
    assert stats["utterances"] == 1 and stats["excluded"] == {"latin_script": 1}
    row = json.loads((tmp_path / "out" / "cv_test.jsonl").read_text())
    assert row["text"] == to_training_text("سال ۱۴۰۲ آمریكا، می رود.")
    assert "آمریکا" in row["text"] and "هزار و چهارصد و دو" in row["text"]  # Persian kaf, spoken number
    assert row["target_lang"] == "fa-IR" and row["speaker"] == "s1"
    info = sf.info(row["audio_filepath"])
    assert (info.samplerate, info.channels) == (16000, 1)
    assert row["duration"] == pytest.approx(2.0, abs=0.05)


def test_fleurs_columns_match_identified_layout():
    assert bes.FLEURS_COLUMNS == ("id", "file", "raw", "normalized", "chars", "num_samples", "gender")
