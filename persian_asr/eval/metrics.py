"""WER/CER for Persian, always computed on `to_scoring_text` of both sides.

Three numbers, because each one hides a different failure:
  - wer:          the headline metric, comparable with published Persian results
                  only when they use the same normalization (say so when quoting).
  - cer:          robust to word-boundary disagreements and long compounds.
  - cer_nospace:  CER with all spaces removed -- isolates spelling errors from
                  segmentation errors ("کتاب خانه" vs "کتابخانه").
"""

from dataclasses import dataclass

import jiwer

from persian_asr.text.asr_text import to_scoring_text


@dataclass(frozen=True)
class Scores:
    wer: float
    cer: float
    cer_nospace: float
    n_utts: int
    n_ref_words: int


def score(refs: list[str], hyps: list[str]) -> Scores:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    pairs = [(to_scoring_text(r), to_scoring_text(h)) for r, h in zip(refs, hyps)]
    # jiwer refuses empty references; an utterance that normalizes to nothing
    # has no words to get right and is dropped from scoring.
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        raise ValueError("no non-empty references after normalization")
    r, h = map(list, zip(*pairs))
    return Scores(
        wer=jiwer.wer(r, h),
        cer=jiwer.cer(r, h),
        cer_nospace=jiwer.cer([x.replace(" ", "") for x in r], [x.replace(" ", "") or " " for x in h]),
        n_utts=len(r),
        n_ref_words=sum(len(x.split()) for x in r),
    )
