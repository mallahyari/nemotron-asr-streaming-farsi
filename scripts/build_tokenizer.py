"""Build the Persian SentencePiece tokenizer for NeMo, then verify it end to end.

    uv run python scripts/build_tokenizer.py build \\
        --train /mnt/asr/manifests/train.jsonl \\
        --check /mnt/asr/manifests/dev.jsonl --check /mnt/asr/manifests/test.jsonl \\
        --out-dir /mnt/asr/tokenizer

Design (each point checked against NeMo's source):

  - Built with NeMo's own `create_spt_model`, with the exact arguments NeMo's
    `scripts/tokenizers/process_asr_text_tokenizer.py` passes for
    `--tokenizer spe --spe_type unigram --no_lower_case
     --spe_user_defined_symbols <ZWNJ>`. NeMo therefore writes `tokenizer.model`,
    `tokenizer.vocab` and `vocab.txt` in the layout `change_vocabulary` loads.
  - Unigram + `nmt_nfkc` normalization, the same as Nemotron 3.5's tokenizer.
    `--no_lower_case` matters: NeMo's default switches to `nmt_nfkc_cf`.
  - ZWNJ (U+200C) is a USER_DEFINED symbol. Without that, `nmt_nfkc` turns it
    into a space and every half-space is lost on the way through the model.
  - Persian only, no language tag in the text: the prompt dataset takes the
    language from the manifest's `target_lang` (`fa-IR`), not from the text.
  - Trained on the TRAIN split only; dev/test are only used to verify.

After building, `verify` (also run automatically) fails loudly unless:
the model has the expected type/normalizer/size, ZWNJ is a user-defined
piece, no transcript in any split produces <unk>, every transcript decodes
back to exactly itself, and NeMo's own SentencePieceTokenizer wrapper gives
the same ids and text as SentencePiece.
"""

import json
from pathlib import Path

import typer
from typing_extensions import Annotated

from persian_asr.text.normalize_fa import ZWNJ

app = typer.Typer(pretty_exceptions_show_locals=False)

# sentencepiece_model.proto enums
UNIGRAM = 1
NORMAL, UNKNOWN, CONTROL, USER_DEFINED = 1, 2, 3, 4
# FastConformer subsamples 8x from 10 ms features: one encoder frame per 80 ms.
ENCODER_FRAME_SEC = 0.08


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def create(train_rows: list[dict], out_dir: Path, vocab_size: int) -> Path:
    """NeMo's create_spt_model with process_asr_text_tokenizer.py's arguments."""
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model

    if out_dir.exists() and any(out_dir.iterdir()):
        # create_spt_model silently returns an existing tokenizer.model instead of rebuilding
        raise SystemExit(f"{out_dir} is not empty; refusing to reuse a stale tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Same corpus file process_asr_text_tokenizer.py builds: one `text` per line.
    corpus = out_dir / "text_corpus" / "document.txt"
    corpus.parent.mkdir()
    with open(corpus, "w") as f:
        for r in train_rows:
            f.write(r["text"] + "\n")
    model_path, vocab_path = create_spt_model(
        data_file=str(corpus),
        vocab_size=vocab_size,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(out_dir),
        tokenizer_type="unigram",
        character_coverage=1.0,
        train_extremely_large_corpus=False,
        max_sentencepiece_length=-1,
        split_by_unicode_script=True,
        bos=False,
        eos=False,
        pad=False,
        control_symbols=None,
        user_defined_symbols=[ZWNJ],
        byte_fallback=False,
        split_digits=False,
        remove_extra_whitespaces=False,
    )
    for name in ("tokenizer.model", "tokenizer.vocab", "vocab.txt"):
        if not (out_dir / name).is_file():
            raise SystemExit(f"NeMo did not write {name}")
    return Path(model_path)


def check_model(model_path: Path, vocab_size: int) -> list[str]:
    """Structural checks on the SentencePiece model proto; returns problems."""
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(model_path.read_bytes())
    problems = []
    if m.trainer_spec.model_type != UNIGRAM:
        problems.append(f"model_type {m.trainer_spec.model_type}, expected unigram ({UNIGRAM})")
    if m.normalizer_spec.name != "nmt_nfkc":
        problems.append(f"normalizer {m.normalizer_spec.name!r}, expected 'nmt_nfkc' (Nemotron's)")
    if len(m.pieces) != vocab_size:
        problems.append(f"{len(m.pieces)} pieces, expected {vocab_size}")
    if m.trainer_spec.byte_fallback:
        problems.append("byte_fallback is on")
    if (m.trainer_spec.unk_id, m.trainer_spec.bos_id, m.trainer_spec.eos_id) != (0, -1, -1):
        problems.append(f"unk/bos/eos ids {m.trainer_spec.unk_id}/{m.trainer_spec.bos_id}/{m.trainer_spec.eos_id}, expected 0/-1/-1")
    types = {p.piece: p.type for p in m.pieces}
    if types.get(ZWNJ) != USER_DEFINED:
        problems.append(f"ZWNJ is not a USER_DEFINED piece (type {types.get(ZWNJ)})")
    if (n_user := sum(t == USER_DEFINED for t in types.values())) != 1:
        problems.append(f"{n_user} USER_DEFINED pieces, expected exactly 1 (ZWNJ)")
    return problems


def check_texts(model_path: Path, rows: list[dict]) -> tuple[list[str], dict]:
    """Every transcript must encode without <unk> and decode back to itself."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    problems, unk_rows, bad_roundtrip = [], [], []
    n_tokens, per_sec, over_frames = 0, [], 0
    for r in rows:
        ids = sp.encode(r["text"])
        if sp.unk_id() in ids:
            unk_rows.append(r["text"])
        if sp.decode(ids) != r["text"]:
            bad_roundtrip.append(r["text"])
        n_tokens += len(ids)
        per_sec.append(len(ids) / r["duration"])
        over_frames += len(ids) > r["duration"] / ENCODER_FRAME_SEC
    if unk_rows:
        problems.append(f"{len(unk_rows)} transcripts produce <unk>, e.g. {unk_rows[:3]}")
    if bad_roundtrip:
        problems.append(f"{len(bad_roundtrip)} transcripts don't decode back to themselves, e.g. {bad_roundtrip[:3]}")
    per_sec.sort()
    stats = {
        "transcripts": len(rows),
        "tokens_per_transcript": round(n_tokens / len(rows), 2),
        "tokens_per_second_median": round(per_sec[len(per_sec) // 2], 2),
        "tokens_per_second_p99": round(per_sec[int(len(per_sec) * 0.99)], 2),
        "tokens_per_second_max": round(per_sec[-1], 2),
        "transcripts_with_more_tokens_than_encoder_frames": over_frames,
    }
    return problems, stats


def check_nemo_wrapper(model_path: Path, rows: list[dict]) -> list[str]:
    """NeMo's SentencePieceTokenizer must agree with SentencePiece exactly."""
    import sentencepiece as spm
    from nemo.collections.common.tokenizers import SentencePieceTokenizer

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    nt = SentencePieceTokenizer(model_path=str(model_path))
    problems = []
    for r in rows:
        ids = nt.text_to_ids(r["text"])
        if ids != sp.encode(r["text"]) or nt.ids_to_text(ids) != r["text"]:
            problems.append(f"NeMo wrapper disagrees on {r['text']!r}")
            break
    if nt.vocab_size != sp.get_piece_size():
        problems.append(f"NeMo vocab_size {nt.vocab_size} != {sp.get_piece_size()}")
    return problems


@app.command()
def verify(
    model: Annotated[Path, typer.Option(help="tokenizer.model to verify")],
    check: Annotated[list[Path], typer.Option(help="manifests whose transcripts must tokenize cleanly")],
    vocab_size: int = 1024,
    nemo: Annotated[bool, typer.Option(help="also check NeMo's SentencePieceTokenizer wrapper")] = True,
) -> dict:
    report = {"model": str(model), "problems": check_model(model, vocab_size), "splits": {}}
    for path in check:
        rows = read_rows([path])
        problems, stats = check_texts(model, rows)
        report["problems"] += [f"{path.name}: {p}" for p in problems]
        report["splits"][path.name] = stats
        if nemo:
            report["problems"] += [f"{path.name}: {p}" for p in check_nemo_wrapper(model, rows[:20000])]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["problems"]:
        raise SystemExit(f"tokenizer verification FAILED: {len(report['problems'])} problem(s)")
    print("tokenizer verification passed")
    return report


@app.command()
def build(
    train: Annotated[Path, typer.Option(help="train manifest (the tokenizer is trained on its `text`)")],
    check: Annotated[list[Path], typer.Option(help="extra manifests to verify against, e.g. dev and test")],
    out_dir: Annotated[Path, typer.Option(help="new, empty directory for the tokenizer")],
    vocab_size: int = 1024,
) -> None:
    model_path = create(read_rows([train]), out_dir, vocab_size)
    report = verify(model=model_path, check=[train, *check], vocab_size=vocab_size, nemo=True)
    (out_dir / "tokenizer_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()
