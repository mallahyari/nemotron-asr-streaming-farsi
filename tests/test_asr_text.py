import pytest

from persian_asr.eval.metrics import score
from persian_asr.text.asr_text import to_scoring_text, to_training_text
from persian_asr.text.normalize_fa import ZWNJ


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Arabic letter forms fold onto Persian ones, punctuation goes.
        ("كتاب علي؟", "کتاب علی"),
        # Verb prefix: space and ZWNJ variants converge on ZWNJ.
        ("او می رود.", f"او می{ZWNJ}رود"),
        (f"او می{ZWNJ}رود", f"او می{ZWNJ}رود"),
        ("نمی دانم", f"نمی{ZWNJ}دانم"),
        # Plural suffix as a separate token joins; words merely starting with ها don't.
        ("کتاب ها و کتاب های من", f"کتاب{ZWNJ}ها و کتاب{ZWNJ}های من"),
        ("هادی آمد", "هادی آمد"),
        # Words that merely start with "می" are untouched.
        ("میز و میوه", "میز و میوه"),
        # Glued forms split via the corpus-built lexicon...
        ("نمیدونم میشه یا نه", f"نمی{ZWNJ}دونم می{ZWNJ}شه یا نه"),
        ("بچهها", f"بچه{ZWNJ}ها"),
        # ...but real words that look like affixed ones stay whole.
        ("تنها میسر میدان", "تنها میسر میدان"),
        # Stretched letters collapse to one; ordinary double letters are kept.
        ("واااااااای بللللله", "وای بله"),
        ("الله ممنون", "الله ممنون"),
        # Collapsing happens before the lexicon, so a stretched glued verb is still split.
        ("میشههههه", f"می{ZWNJ}شه"),
        # Numbers are spoken form.
        ("سال ۱۴۰۲", "سال هزار و چهارصد و دو"),
    ],
)
def test_training_text(raw, expected):
    assert to_training_text(raw) == expected


def test_scoring_text_ignores_zwnj_vs_space():
    assert to_scoring_text("می‌رود") == to_scoring_text("می رود") == "می رود"


def test_score_perfect_and_zwnj_insensitive():
    s = score(["او می رود."], ["او می‌رود"])
    assert s.wer == 0 and s.cer == 0 and s.cer_nospace == 0


def test_score_counts_errors():
    s = score(["کتاب من"], ["کتاب تو"])
    assert s.wer == 0.5
    assert s.n_ref_words == 2


def test_score_drops_empty_refs():
    s = score(["", "سلام"], ["چیزی", "سلام"])
    assert s.n_utts == 1 and s.wer == 0


def test_lexicon_can_be_disabled():
    assert to_training_text("میشه", use_lexicon=False) == "میشه"
