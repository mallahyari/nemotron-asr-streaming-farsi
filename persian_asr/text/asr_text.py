"""Persian transcript forms for ASR: what the model is trained on, and what it is scored on.

Built on the TTS normalizer (`normalize_fa`), which already folds Arabic letter
variants onto Persian ones, drops diacritics, keeps ZWNJ and spells numbers out.
ASR needs two things on top of that:

  - **Training text** (`to_training_text`): no punctuation (v1 labels are too
    noisy to learn it from), and ONE spelling per word. The corpora disagree on
    whether the verb prefix and the plural suffix are joined with a ZWNJ, a
    space, or nothing ("می‌رود" / "می رود" / "میرود"). An ASR model trained on all
    three learns to guess between them, and every guess against a reference
    that chose differently is a word error. Everything is folded onto the
    standard ZWNJ form: the space variant by rule, the glued variant ("میشه")
    through `zwnj_lexicon.json`, because a glued word can't be split by rule
    ("میز", "میوه", "تنها" are real words); the lexicon was mined from the training corpora.

  - **Scoring text** (`to_scoring_text`): additionally maps ZWNJ to a space, so
    "می‌رود" vs "می رود" is never counted as an error -- they are the same words
    spoken aloud. This is the same convention as pocket-tts's `eval_fa.wer_text`.

Spoken-form numbers are a deliberate choice: the audio says "هزار و چهارصد و
دو", so that is the target. Digits, if wanted, are an inverse-normalization step
after recognition.
"""

import json
import re
from functools import cache
from pathlib import Path

from persian_asr.text.normalize_fa import (
    KEPT_PUNCT,
    PERSIAN_LETTERS,
    ZWNJ,
    normalize,
    reject_reason,
)

__all__ = ["to_training_text", "to_scoring_text", "reject_reason"]

LEXICON_PATH = Path(__file__).with_name("zwnj_lexicon.json")

_PUNCT_RE = re.compile(f"[{re.escape(KEPT_PUNCT)}]")
_L = f"[{PERSIAN_LETTERS}]"
# "می رود" / "نمی رود" -> joined with ZWNJ. Only when "می"/"نمی" is a whole
# token (preceded by start or space) and followed by another word.
_VERB_PREFIX_RE = re.compile(rf"(?:(?<=\s)|^)(ن?می) (?={_L})")
# "کتاب ها" / "کتاب های" / "کتاب هایی" -> joined with ZWNJ. Only when the
# suffix is a whole token, so words like "هادی" or "هایده" are untouched.
_PLURAL_SUFFIX_RE = re.compile(rf"(?<={_L}) (ها|های|هایی)(?=\s|$)")
_STRETCHED_RE = re.compile(rf"({_L})\1{{2,}}")


def _strip_punct(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", text)).strip()


@cache
def _lexicon() -> dict[str, str]:
    """{glued: ZWNJ-separated}; loaded from zwnj_lexicon.json."""
    if not LEXICON_PATH.exists():
        return {}
    return json.loads(LEXICON_PATH.read_text())


def to_training_text(text: str, *, use_lexicon: bool = True) -> str:
    """Canonical transcript for ASR training manifests (and tokenizer training)."""
    text = _strip_punct(normalize(text))
    # Subtitles spell an elongated vowel or consonant by repeating the letter
    # ("واااااای", "بللللله"); the word spoken is "وای"/"بله". No Persian word
    # has the same letter three times in a row, so any such run is collapsed.
    text = _STRETCHED_RE.sub(r"\1", text)
    text = _VERB_PREFIX_RE.sub(rf"\1{ZWNJ}", text)
    text = _PLURAL_SUFFIX_RE.sub(rf"{ZWNJ}\1", text)
    if use_lexicon and (lex := _lexicon()):
        text = " ".join(lex.get(w, w) for w in text.split(" "))
    return text


def to_scoring_text(text: str) -> str:
    """Form used for WER/CER: training text with ZWNJ read as a word boundary.

    Apply to BOTH reference and hypothesis, whatever system produced them.
    """
    return re.sub(r"\s+", " ", to_training_text(text).replace(ZWNJ, " ")).strip()
