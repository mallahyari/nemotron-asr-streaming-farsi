import os
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gcp" / "fix_checkpoints.py"
LAST = "run--val_wer=0.6{e}00-epoch={e}-last.ckpt"
TOPK = "run--val_wer=0.6{e}00-epoch={e}.ckpt"


def run(d: Path) -> str:
    return subprocess.run([sys.executable, str(SCRIPT), str(d)], check=True, capture_output=True, text=True).stdout


def names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir() if p.is_file()}


def ckpt(path: Path, payload: str = "weights") -> Path:
    # torch.save writes a zip archive; a small real zip stands in for it
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", payload * 1000)
    return path


def truncated(path: Path) -> Path:
    # what a power loss left behind: the first part of a valid checkpoint
    ckpt(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) * 9 // 10])
    return path


def marker(d: Path, name: str) -> None:
    (d / (name.removesuffix(".ckpt") + "-unfinished")).write_text("")


def make(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


def test_missing_dir_is_fresh_start(tmp_path):
    assert "fresh start" in run(tmp_path / "nope")


def test_clean_dir_untouched(tmp_path):
    d = make(tmp_path)
    for n in (LAST.format(e=3), TOPK.format(e=3), TOPK.format(e=2)):
        ckpt(d / n)
    (d / "run.nemo").write_text("model")
    before = names(d)
    out = run(d)
    assert names(d) == before
    assert f"resuming from {LAST.format(e=3)}" in out


def test_unfinished_last_removed_previous_kept(tmp_path):
    # preempted while writing epoch 4's last: the finished epoch-3 one survives
    d = make(tmp_path)
    ckpt(d / LAST.format(e=3))
    truncated(d / LAST.format(e=4))
    marker(d, LAST.format(e=4))
    out = run(d)
    assert names(d) == {LAST.format(e=3)}
    assert f"resuming from {LAST.format(e=3)}" in out


def test_only_unfinished_last_gives_fresh_start(tmp_path):
    d = make(tmp_path)
    truncated(d / LAST.format(e=0))
    marker(d, LAST.format(e=0))
    out = run(d)
    assert names(d) == set()
    assert "fresh start" in out


def test_two_finished_lasts_keep_highest_epoch(tmp_path):
    # e.g. after a resume Lightning keeps the resumed-from last checkpoint
    d = make(tmp_path)
    new, old = ckpt(d / LAST.format(e=10), "new"), ckpt(d / LAST.format(e=9), "old")  # "10" < "9" as text
    os.utime(new, (1, 1))  # mtime must not override epoch order
    out = run(d)
    assert names(d) == {new.name}
    assert (tmp_path / "stale_last" / old.name).exists()
    assert f"resuming from {new.name}" in out


def test_unfinished_topk_removed_nemo_kept(tmp_path):
    d = make(tmp_path)
    ckpt(d / LAST.format(e=5))
    truncated(d / TOPK.format(e=5))
    marker(d, TOPK.format(e=5))
    (d / "run.nemo").write_text("model")
    run(d)
    assert names(d) == {LAST.format(e=5), "run.nemo"}


def test_corrupt_last_falls_back_to_same_step_topk(tmp_path):
    # the real preemption test: epoch-1 last.ckpt finished (no marker) but
    # truncated on disk; the epoch-1 top-k saved just before it is intact and
    # the epoch-0 last was already deleted
    d = make(tmp_path)
    ckpt(d / TOPK.format(e=0))
    good = ckpt(d / TOPK.format(e=1))
    truncated(d / LAST.format(e=1))
    out = run(d)
    link = d / LAST.format(e=1)
    assert (tmp_path / "corrupt" / LAST.format(e=1)).exists()
    assert link.exists() and os.path.samefile(link, good)  # hard link, no copy
    assert names(d) == {TOPK.format(e=0), TOPK.format(e=1), LAST.format(e=1)}
    assert f"resuming from {LAST.format(e=1)} (hard link to top-k {TOPK.format(e=1)})" in out


def test_crc_damage_detected(tmp_path):
    # right size but damaged bytes inside a member: only a CRC check catches it
    d = make(tmp_path)
    ckpt(d / TOPK.format(e=1))
    bad = ckpt(d / LAST.format(e=2))
    data = bytearray(bad.read_bytes())
    i = data.index(b"weights") + 100
    data[i:i + 50] = b"\0" * 50
    bad.write_bytes(bytes(data))
    out = run(d)
    assert (tmp_path / "corrupt" / LAST.format(e=2)).exists()
    assert f"hard link to top-k {TOPK.format(e=1)}" in out


def test_all_corrupt_gives_fresh_start(tmp_path):
    d = make(tmp_path)
    truncated(d / LAST.format(e=1))
    truncated(d / TOPK.format(e=1))
    out = run(d)
    assert names(d) == set()
    assert "no intact checkpoint (fresh start)" in out
