"""Tests for the tokenizer checks. Training uses SentencePiece directly with the
same settings NeMo's create_spt_model passes (that function itself is run and
checked on the GPU VM, where NeMo is installed)."""

import importlib.util
import json
from pathlib import Path

import pytest
import sentencepiece as spm

from persian_asr.text.normalize_fa import ZWNJ

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_tokenizer.py"
spec = importlib.util.spec_from_file_location("build_tokenizer", SCRIPT)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

TEXTS = [f"او می{ZWNJ}خواهد کتاب{ZWNJ}ها را بخواند", "سلام دنیا", f"نمی{ZWNJ}دونم چی شده", "گچ پژواک ژاله چه خوب"]


def train(tmp_path, vocab_size=60, user_symbols=(ZWNJ,), rule="nmt_nfkc", texts=TEXTS):
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(texts * 300))
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(tmp_path / "tokenizer"), vocab_size=vocab_size,
        model_type="unigram", character_coverage=1.0, hard_vocab_limit=False, bos_id=-1, eos_id=-1,
        user_defined_symbols=list(user_symbols), normalization_rule_name=rule,
        remove_extra_whitespaces=False, minloglevel=2,
    )
    return tmp_path / "tokenizer.model"


def rows(texts):
    return [{"text": t, "duration": 2.0} for t in texts]


def test_good_model_passes(tmp_path):
    model = train(tmp_path)
    n = spm.SentencePieceProcessor(model_file=str(model)).get_piece_size()
    assert bt.check_model(model, n) == []
    problems, stats = bt.check_texts(model, rows(TEXTS))
    assert problems == [] and stats["transcripts"] == len(TEXTS)


def test_zwnj_survives_roundtrip(tmp_path):
    sp = spm.SentencePieceProcessor(model_file=str(train(tmp_path)))
    t = f"می{ZWNJ}خواهد"
    assert sp.decode(sp.encode(t)) == t
    assert ZWNJ in sp.id_to_piece(sp.encode(t))


def test_missing_zwnj_symbol_is_caught(tmp_path):
    # Without the user-defined symbol nmt_nfkc turns ZWNJ into a space: both checks must fire.
    model = train(tmp_path, user_symbols=())
    n = spm.SentencePieceProcessor(model_file=str(model)).get_piece_size()
    assert any("ZWNJ is not a USER_DEFINED" in p for p in bt.check_model(model, n))
    problems, _ = bt.check_texts(model, rows(TEXTS))
    assert any("don't decode back" in p for p in problems)


def test_wrong_normalizer_and_size_are_caught(tmp_path):
    model = train(tmp_path, rule="nmt_nfkc_cf")
    problems = bt.check_model(model, 999)
    assert any("normalizer" in p for p in problems) and any("pieces, expected 999" in p for p in problems)


def test_unknown_characters_are_caught(tmp_path):
    model = train(tmp_path)
    problems, _ = bt.check_texts(model, rows(["سلام ڤ"]))  # ڤ never seen in training
    assert any("<unk>" in p for p in problems)


def test_tokens_per_second_and_frame_budget(tmp_path):
    model = train(tmp_path)
    sp = spm.SentencePieceProcessor(model_file=str(model))
    n = len(sp.encode(TEXTS[0]))
    # 0.08 s clip = 1 encoder frame; any transcript with >1 token exceeds it
    _, stats = bt.check_texts(model, [{"text": TEXTS[0], "duration": 0.08}])
    assert stats["transcripts_with_more_tokens_than_encoder_frames"] == (1 if n > 1 else 0)
    assert stats["tokens_per_second_max"] == pytest.approx(n / 0.08, abs=0.01)
