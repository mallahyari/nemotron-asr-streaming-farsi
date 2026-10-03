"""Run NeMo's OOMptimizer on Nemotron 3.5 + our Persian tokenizer.

    python scripts/oomptimizer_prompt.py --base /models/nemotron-3.5-asr-streaming-0.6b.nemo \\
        --tokenizer-dir /mnt/asr/tokenizer --out /models/nemotron-fa-tokenizer.nemo \\
        -- --buckets '[1.6,2.3,...,20.0]'          # everything after -- goes to OOMptimizer

Why a wrapper: NeMo v3.0.0's `scripts/speech_recognition/oomptimizer.py` builds
synthetic batches from `model.oomptimizer_schema`. EncDecRNNTBPEModelWithPrompt
doesn't define one and inherits ASRModel's 4-element schema (audio, audio_len,
tokens, token_len), but its `training_step` unpacks 5 (+ prompt_indices).
The schema generator has no way to emit a per-sample (B,) index tensor.

So this wrapper (1) builds the model we will actually train -- the base
checkpoint after `change_vocabulary(<our tokenizer>, "bpe")`, exactly what
speech_to_text_finetune.py's update_tokenizer does -- and saves it as a .nemo;
(2) wraps `training_step` to append prompt_indices = fa-IR (38) for every sample
when given a 4-tuple (GPU memory doesn't depend on which index is used); and
(3) runs NeMo's unmodified OOMptimizer command on that .nemo.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

NEMO_SRC = Path("/opt/nemo-src")


def main() -> None:
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="fa-IR")
    a = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)

    import torch
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

    if not Path(a.out).exists():
        model = ASRModel.restore_from(a.base, map_location="cpu")
        model.change_vocabulary(new_tokenizer_dir=a.tokenizer_dir, new_tokenizer_type="bpe")
        assert model.tokenizer.vocab_size == 1024, model.tokenizer.vocab_size
        model.save_to(a.out)
        print(f"saved {a.out} (vocab {model.tokenizer.vocab_size}, fuse_loss_wer={model.joint.fuse_loss_wer})")
        del model
    pd_index = ASRModel.restore_from(a.out, return_config=True).model_defaults.prompt_dictionary[a.prompt]
    print(f"prompt {a.prompt} -> index {pd_index}")

    original = EncDecRNNTBPEModelWithPrompt.training_step

    def training_step_with_prompt(self, batch, batch_nb):
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            signal = batch[0]
            prompt = torch.full((signal.shape[0],), pd_index, dtype=torch.long, device=signal.device)
            batch = (*batch, prompt)
        return original(self, batch, batch_nb)

    EncDecRNNTBPEModelWithPrompt.training_step = training_step_with_prompt

    spec = importlib.util.spec_from_file_location("oomptimizer", NEMO_SRC / "scripts/speech_recognition/oomptimizer.py")
    oom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oom)
    oom.oomptimizer.main(args=["--pretrained-name", a.out, *rest], standalone_mode=True)


if __name__ == "__main__":
    main()
