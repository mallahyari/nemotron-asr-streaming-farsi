"""Evaluate a NeMo ASR model on our evaluation manifests (WER / CER / space-free CER).

    uv run python scripts/evaluate.py --model nvidia/stt_fa_fastconformer_hybrid_large --decoder ctc \\
        --manifest /mnt/asr/eval/fleurs_test.jsonl --manifest /mnt/asr/eval/cv_test.jsonl \\
        --manifest /mnt/asr/manifests/dev.jsonl --manifest /mnt/asr/manifests/test.jsonl \\
        --out /mnt/asr/results/stt_fa_ctc --nemo-dir ~/nemo-v3.0.0

Follows NVIDIA's evaluation contract (NeMo Speech skill `nemo-speech-asr-finetune`,
references/evaluation-style-contract.md):

  1. Transcribe with NeMo's own `examples/asr/speech_to_text_eval.py` from the
     checkout matching the installed NeMo version, with `amp=false
     compute_dtype=bfloat16 matmul_precision=high` and greedy_batch decoding.
  2. Score a DERIVED manifest in which both `text` and `pred_text` pass through
     the same documented normalizer, the contract's option for language-specific
     normalization: `persian_asr.text.asr_text.to_scoring_text` (ZWNJ read as a
     space, punctuation removed, Arabic letter forms folded, numbers spelled out).

Our WER is cross-checked against NeMo's own `word_error_rate` on the same
normalized pairs; the script fails if they disagree. Raw predictions are kept.

Language-prompt models (Nemotron 3.5 streaming, EncDecRNNTBPEModelWithPrompt):
pass `--target-lang fa-IR`. NeMo's eval path would otherwise pick the prompt at
random per utterance ("auto" half the time); scripts/nemo_eval_prompt.py runs
the same NeMo script with the prompt fixed, and we check it covered every
utterance. For cache-aware streaming models, `--extra att_context_size=[56,13]`
picks the look-ahead (default: the model's first, here [56,3]).

`--streaming` (cache-aware streaming models): transcribe with NeMo's
`examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`
instead, which feeds the audio chunk by chunk with the encoder caches, as a live
stream would. That script sets the prompt itself (`target_lang`, via
`set_inference_prompt`) and only supports float32. Its output has no audio
paths, so rows are matched to the manifest by position and their reference
text must match exactly. Scoring is identical.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from typing_extensions import Annotated

from persian_asr.eval.metrics import score
from persian_asr.text.asr_text import to_scoring_text

app = typer.Typer(pretty_exceptions_show_locals=False)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def nemo_transcribe(nemo_dir: Path, model: str, manifest: Path, out_file: Path, decoder: str | None,
                    batch_size: int, extra: list[str], target_lang: str | None = None) -> dict | None:
    script = nemo_dir / "examples" / "asr" / "speech_to_text_eval.py"
    if not script.is_file():
        raise SystemExit(f"{script} not found; clone NeMo at the installed version (e.g. --branch v3.0.0)")
    model_arg = f"model_path={model}" if model.endswith(".nemo") else f"pretrained_name={model}"
    cmd = [sys.executable, str(script), model_arg, f"dataset_manifest={manifest}", f"output_filename={out_file}",
           f"batch_size={batch_size}", "amp=false", "compute_dtype=bfloat16", "matmul_precision=high",
           "use_cer=false"]
    if decoder == "ctc":
        cmd += ["decoder_type=ctc", "ctc_decoding.strategy=greedy_batch"]
    elif decoder == "rnnt":
        cmd += ["decoder_type=rnnt", "rnnt_decoding.strategy=greedy_batch",
                "rnnt_decoding.greedy.use_cuda_graph_decoder=true"]
    cmd += extra
    count_file = None
    if target_lang:
        count_file = out_file.with_suffix(".prompt.json")
        count_file.unlink(missing_ok=True)
        wrapper = Path(__file__).resolve().parent / "nemo_eval_prompt.py"
        cmd = [sys.executable, str(wrapper), target_lang, str(count_file), *cmd[1:]]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(count_file.read_text()) if count_file else None


def nemo_stream_transcribe(nemo_dir: Path, model: str, manifest: Path, out: Path, batch_size: int,
                          extra: list[str], target_lang: str | None) -> list[dict]:
    script = nemo_dir / "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
    if not script.is_file():
        raise SystemExit(f"{script} not found")
    stream_dir = out / f"streaming_{manifest.stem}"
    shutil.rmtree(stream_dir, ignore_errors=True)
    model_arg = f"model_path={model}" if model.endswith(".nemo") else f"pretrained_name={model}"
    cmd = [sys.executable, str(script), model_arg, f"dataset_manifest={manifest}", f"output_path={stream_dir}",
           f"batch_size={batch_size}", "amp=false", "compute_dtype=float32", "matmul_precision=high"]
    if target_lang:
        cmd.append(f"target_lang={target_lang}")
    cmd += extra
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    outs = sorted(stream_dir.glob("streaming_out_*.json"))
    if len(outs) != 1:
        raise SystemExit(f"{stream_dir}: expected one streaming_out_*.json, found {outs}")
    return read_jsonl(outs[0])


@app.command()
def main(
    model: Annotated[str, typer.Option(help="pretrained name or path to a .nemo")],
    manifest: Annotated[list[Path], typer.Option(help="eval manifest(s)")],
    out: Annotated[Path, typer.Option(help="output directory")],
    nemo_dir: Annotated[Path, typer.Option(help="NeMo checkout matching the installed version")],
    decoder: Annotated[str | None, typer.Option(help="hybrid models: ctc or rnnt")] = None,
    batch_size: int = 32,
    extra: Annotated[list[str] | None, typer.Option(help="extra Hydra overrides for the NeMo script")] = None,
    target_lang: Annotated[str | None, typer.Option(help="language-prompt models: fixed prompt, e.g. fa-IR")] = None,
    streaming: Annotated[bool, typer.Option(help="cache-aware streaming inference (chunk by chunk)")] = False,
) -> None:
    from nemo.collections.asr.metrics.wer import word_error_rate

    out.mkdir(parents=True, exist_ok=True)
    results = {"model": model, "decoder": decoder,
               "nemo_script": "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
               if streaming else "examples/asr/speech_to_text_eval.py",
               "precision": "amp=false compute_dtype=float32" if streaming else "amp=false compute_dtype=bfloat16",
               "streaming": streaming, "target_lang": target_lang,
               "extra": extra or [], "sets": {}}
    for path in manifest:
        name = path.stem
        pred_path = out / f"predictions_{name}.json"
        src = read_jsonl(path)
        if streaming:
            rows = nemo_stream_transcribe(nemo_dir, model, path, out, batch_size, extra or [], target_lang)
            if len(rows) != len(src) or any(r["text"] != s["text"] for r, s in zip(rows, src)):
                raise SystemExit(f"{name}: streaming output doesn't line up with {path}")
            preds = [{"audio_filepath": s["audio_filepath"], "pred_text": r["pred_text"]} for r, s in zip(rows, src)]
            with open(pred_path, "w") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            prompt = None
        else:
            prompt = nemo_transcribe(nemo_dir, model, path, pred_path, decoder, batch_size, extra or [], target_lang)
            preds = read_jsonl(pred_path)
        if prompt is not None:
            if prompt["utterances"] != len(src):
                raise SystemExit(f"{name}: fixed prompt applied to {prompt['utterances']} of {len(src)} utterances")
            print(f"{name}: prompt {target_lang} (index {prompt['prompt_index']}) on all {len(src)} utterances; "
                  f"dataset would have used {prompt['dataset_prompt_indices']}", flush=True)
            results.setdefault("prompt", {})[name] = prompt
        if len(preds) != len(src) or any(p["audio_filepath"] != s["audio_filepath"] for p, s in zip(preds, src)):
            raise SystemExit(f"{pred_path}: predictions don't line up with {path}")

        # derived, normalized manifest (kept for inspection) + scores
        norm = [{"id": s.get("id"), "text": to_scoring_text(s["text"]), "pred_text": to_scoring_text(p["pred_text"]),
                 "source": s.get("source", "")} for s, p in zip(src, preds)]
        with open(out / f"scored_{name}.jsonl", "w") as f:
            for r in norm:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results["sets"][name] = {}
        sources = sorted({r["source"] for r in norm})
        groups = {"all": norm} | ({s: [r for r in norm if r["source"] == s] for s in sources} if len(sources) > 1 else {})
        for g, rows in groups.items():
            s = score([r["text"] for r in rows], [r["pred_text"] for r in rows])
            kept = [r for r in rows if r["text"]]
            nemo_wer = word_error_rate([r["pred_text"] for r in kept], [r["text"] for r in kept])
            if abs(nemo_wer - s.wer) > 1e-9:
                raise SystemExit(f"{name}/{g}: our WER {s.wer} != NeMo word_error_rate {nemo_wer}")
            results["sets"][name][g] = {"wer": round(s.wer, 4), "cer": round(s.cer, 4), "cer_nospace": round(s.cer_nospace, 4),
                                        "utterances": s.n_utts, "ref_words": s.n_ref_words}
            print(f"{name:<14} {g:<13} WER {s.wer:6.2%}  CER {s.cer:6.2%}  CER(no space) {s.cer_nospace:6.2%}  n={s.n_utts}")
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()
