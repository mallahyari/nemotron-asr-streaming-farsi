"""Split a manifest into shuffled shards for Lhotse training.

    uv run python scripts/shard_manifest.py /mnt/asr/manifests/train.jsonl /mnt/asr/shards/train --per-shard 200

Writes manifest_0.json .. manifest_{N-1}.json and prints the NeMo path pattern
to pass as `manifest_filepath` (manifest__OP_0..{N-1}_CL_.json).

NVIDIA's NeMo fine-tuning guidance: shard training manifests (~200 utterances
per shard) even when not tarred, so a restart after preemption doesn't begin
iterating from the start of one giant manifest. Rows are shuffled with a fixed
seed first, so every shard mixes sources and the split is reproducible. The
union of the shards is exactly the input (checked).
"""

import json
import math
import random
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    manifest: Path,
    out_dir: Path,
    per_shard: Annotated[int, typer.Option()] = 200,
    seed: int = 0,
) -> None:
    lines = [l for l in manifest.read_text().splitlines() if l.strip()]
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = lines[:]
    rng.shuffle(order)
    n = math.ceil(len(order) / per_shard)
    for i in range(n):
        (out_dir / f"manifest_{i}.json").write_text("\n".join(order[i * per_shard:(i + 1) * per_shard]) + "\n")
    # union of shards == input, no duplicates or losses
    back = [l for i in range(n) for l in (out_dir / f"manifest_{i}.json").read_text().splitlines() if l.strip()]
    if sorted(back) != sorted(lines):
        raise SystemExit("shards don't reproduce the input manifest")
    hours = sum(json.loads(l)["duration"] for l in lines) / 3600
    print(f"{len(lines):,} rows ({hours:,.1f} h) -> {n} shards in {out_dir}")
    print(f"manifest_filepath={out_dir}/manifest__OP_0..{n - 1}_CL_.json")


if __name__ == "__main__":
    app()
