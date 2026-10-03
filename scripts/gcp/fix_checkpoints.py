"""Make a NeMo checkpoint dir resumable before (re)launching training.

    python3 scripts/gcp/fix_checkpoints.py ~/exp/<EXP_NAME>/checkpoints

Standard library only (runs on the VM host, outside the container).

NeMo v3.0.0's exp_manager (`check_resume`) resumes only from `*last.ckpt`, and
fails in states a spot preemption leaves behind:

1. The only `*last.ckpt` is *unfinished* (NeMoModelCheckpoint writes a
   `<ckpt minus .ckpt>-unfinished` marker before saving or removing a
   checkpoint, and deletes it afterwards): "Last checkpoint is unfinished and
   cannot be used to resume".
2. Several finished `*last.ckpt`: "Multiple checkpoints ... *last.ckpt".
   Lightning saves the new last checkpoint before removing the previous one,
   and never removes the checkpoint it resumed from (`_should_remove_checkpoint`),
   so after any resume two last checkpoints stay around.
3. A *finished but corrupt* checkpoint: torch.save returned and the marker was
   removed, but the data was still in the page cache when the VM lost power.
   This happens in practice: a last.ckpt was truncated (6.7 of 7.5 GB) and
   torch.load failed with "failed finding central directory".

Steps: (a) delete every *.ckpt whose marker exists, then the markers (what
NeMo's own `_remove_unfinished_checkpoints` does); (b) walk the remaining
*.ckpt newest first -- highest `epoch=N` (each "epoch" is VAL_EVERY steps
here), last before top-k, then mtime -- and keep the first that is an intact
zip with all CRCs correct (torch.save writes a zip); corrupt ones met on the
way go to `<exp>/corrupt/`; (c) move all other *last.ckpt to
`<exp>/stale_last/`; (d) if the survivor is a top-k checkpoint (saved at the
same step, same state), hard-link it as `<name>-last.ckpt` so NeMo picks it up.
The moved-to dirs are outside the checkpoint dir because NeMo searches it with
rglob.
"""

import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

SUFFIX = "-unfinished"  # NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX


def marker_for(ckpt: Path) -> Path:
    # NeMoModelCheckpoint.format_checkpoint_unfinished_marker_path (non model-parallel)
    s = str(ckpt).removesuffix(".nemo").removesuffix(".ckpt").removesuffix("-EMA")
    return Path(s + SUFFIX)


def epoch_of(path: Path) -> int:
    m = re.search(r"epoch=(\d+)", path.name)
    return int(m.group(1)) if m else -1


def is_last(path: Path) -> bool:
    return path.name.endswith("last.ckpt")


def intact(path: Path) -> bool:
    """True if `path` is a readable zip whose members all match their CRC-32."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def fix(ckpt_dir: Path) -> list[str]:
    actions: list[str] = []
    if not ckpt_dir.is_dir():
        return [f"no checkpoint dir {ckpt_dir} (fresh start)"]

    # (a) unfinished
    markers = {m.resolve() for m in ckpt_dir.glob(f"*{SUFFIX}") if m.is_file()}
    for ckpt in sorted(ckpt_dir.rglob("*.ckpt")):
        if ckpt.is_file() and marker_for(ckpt.resolve()) in markers:
            ckpt.unlink()
            actions.append(f"removed unfinished {ckpt.name}")
    for d in sorted(p for p in ckpt_dir.glob("*") if p.is_dir()):
        if marker_for(d.resolve()) in markers:
            shutil.rmtree(d)
            actions.append(f"removed unfinished dir {d.name}")
    for m in sorted(markers):
        m.unlink()
        actions.append(f"removed marker {m.name}")

    # (b) newest intact checkpoint
    ckpts = sorted((p for p in ckpt_dir.rglob("*.ckpt") if p.is_file()),
                   key=lambda p: (epoch_of(p), is_last(p), p.stat().st_mtime), reverse=True)
    chosen = None
    for p in ckpts:
        if intact(p):
            chosen = p
            break
        corrupt = ckpt_dir.parent / "corrupt"
        corrupt.mkdir(exist_ok=True)
        shutil.move(str(p), str(corrupt / p.name))
        actions.append(f"CORRUPT {p.name} -> {corrupt}")

    # (c) one last checkpoint only
    for p in [p for p in ckpt_dir.rglob("*last.ckpt") if p.is_file() and p != chosen]:
        stale = ckpt_dir.parent / "stale_last"
        stale.mkdir(exist_ok=True)
        shutil.move(str(p), str(stale / p.name))
        actions.append(f"moved other last checkpoint {p.name} -> {stale}")

    # (d) resume target
    if chosen is None:
        actions.append("no intact checkpoint (fresh start)")
    elif is_last(chosen):
        actions.append(f"resuming from {chosen.name}")
    else:
        link = chosen.with_name(chosen.name.removesuffix(".ckpt") + "-last.ckpt")
        os.link(chosen, link)
        actions.append(f"resuming from {link.name} (hard link to top-k {chosen.name})")
    return actions


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    for a in fix(Path(sys.argv[1]).expanduser()):
        print(f"fix_checkpoints: {a}", flush=True)


if __name__ == "__main__":
    main()
