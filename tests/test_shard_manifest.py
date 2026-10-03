import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shard_manifest.py"


def test_shards_partition_the_input(tmp_path):
    rows = [{"id": f"c{i}", "duration": 1.0} for i in range(1005)]
    m = tmp_path / "train.jsonl"
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = subprocess.run([sys.executable, str(SCRIPT), str(m), str(tmp_path / "shards"), "--per-shard", "200"],
                         check=True, capture_output=True, text=True).stdout
    shards = sorted((tmp_path / "shards").iterdir())
    assert len(shards) == 6  # 5 x 200 + 1 x 5
    ids = [json.loads(l)["id"] for s in shards for l in s.read_text().splitlines()]
    assert sorted(ids) == sorted(r["id"] for r in rows) and len(ids) == len(set(ids))
    assert "manifest__OP_0..5_CL_.json" in out
    first = [json.loads(l)["id"] for l in (tmp_path / "shards" / "manifest_0.json").read_text().splitlines()]
    assert first != [f"c{i}" for i in range(200)]  # shuffled, not the input order
