"""Run NeMo's speech_to_text_eval.py on a prompt model with ONE fixed language prompt.

    python scripts/nemo_eval_prompt.py fa-IR <count_file> /opt/nemo-src/examples/asr/speech_to_text_eval.py \\
        model_path=... dataset_manifest=... output_filename=... [more Hydra overrides]

Called by scripts/evaluate.py (--target-lang); not meant to be run by hand.

Why a wrapper (NeMo Speech v3.0.0, EncDecRNNTBPEModelWithPrompt):
- speech_to_text_eval.py -> transcribe_speech.py has no option for this model's
  language prompt (its `prompt` field is for Canary-style multitask models).
- The model's transcribe dataloader (`_setup_transcribe_dataloader`) builds a
  LhotseSpeechToTextBpeDatasetWithPromptIndex without `default_prompt_mode`,
  so the dataset uses its default "unified": each utterance gets the "auto"
  prompt with probability 0.5 and its language ID otherwise -- random and not
  what we trained (we fine-tuned with prompt_mode langID, fa-IR only).
- `_transcribe_forward` uses those per-batch prompt indices when present and
  only falls back to `trcfg.target_lang` (default "auto") when they're absent.

So we patch `_transcribe_forward`, the one place every transcription batch
passes through, to always use prompt_dictionary[<lang>], and count the
utterances it saw. The count (plus how many would have been "auto") is written
to <count_file>; evaluate.py checks it equals the manifest size. Everything else
is NVIDIA's unmodified script, run as __main__.
"""

import atexit
import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    lang, count_file, script, *hydra_args = sys.argv[1:]

    import torch
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

    stats = {"target_lang": lang, "prompt_index": None, "utterances": 0, "dataset_prompt_indices": {}}
    original = EncDecRNNTBPEModelWithPrompt._transcribe_forward

    def forced(self, batch, trcfg):
        idx = self.cfg.model_defaults.prompt_dictionary[lang]
        audio = batch[0]
        n = audio.shape[0]
        if len(batch) >= 5 and batch[4] is not None:  # what the dataset would have used
            for v in batch[4].tolist():
                stats["dataset_prompt_indices"][str(v)] = stats["dataset_prompt_indices"].get(str(v), 0) + 1
        prompt = torch.full((n,), idx, dtype=torch.long, device=audio.device)
        rest = list(batch[2:4]) + [None] * (2 - len(batch[2:4]))
        stats["prompt_index"] = idx
        stats["utterances"] += n
        return original(self, (batch[0], batch[1], *rest, prompt), trcfg)

    EncDecRNNTBPEModelWithPrompt._transcribe_forward = forced
    atexit.register(lambda: Path(count_file).write_text(json.dumps(stats) + "\n"))

    sys.path.insert(0, str(Path(script).parent))  # speech_to_text_eval imports transcribe_speech
    sys.argv = [script, *hydra_args]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
