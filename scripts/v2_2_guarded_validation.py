"""Validation pilot of concise grouping, deterministic gates, and one retry.

All cases are from fold-1 validation. The frozen test protocol is untouched.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cloud_prepare import prepare
from scripts.evaluate_v2_2 import load_model
from src.data.v2_2_dataset import load_samples
from src.io_utils import file_hash, strict_loads, write_json, write_jsonl
from src.report.v2_2_guarded import generate_guarded
from src.report.v2_2_grouped_template import render_grouped_from_prompt


MAX_NEW_TOKENS = 1024


def make_generator(model, tokenizer, repetition_penalty, no_repeat_ngram_size):
    import torch

    def generate(prompt):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
        encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        seconds = time.monotonic() - start
        tokens = output.shape[-1] - encoded["input_ids"].shape[-1]
        raw = tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:],
                               skip_special_tokens=True)
        return {"raw_output": raw, "tokens": tokens, "seconds": seconds}

    return generate


def run(annotations_root, adapter_root, protocol_path, output, limit=10,
        repetition_penalty=1.0, no_repeat_ngram_size=6):
    if not 1 <= limit <= 46:
        raise ValueError("This pilot must use 1-46 validation cases")
    if repetition_penalty < 1 or no_repeat_ngram_size < 0:
        raise ValueError("Invalid repetition controls")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    protocol = strict_loads(Path(protocol_path).read_text(encoding="utf-8"))
    adapter_root = Path(adapter_root).resolve()
    if file_hash(adapter_root / "adapter_model.safetensors") != protocol["adapter_sha256"]:
        raise ValueError("Adapter differs from frozen baseline protocol")
    output.mkdir(parents=True)
    prepare(annotations_root, output / "derived",
            source_versions={"annotations": protocol["annotations_dataset"]})
    patients = output / "derived/dataset/dataset_patients.jsonl"
    master = output / "derived/dataset/dataset_master.csv"
    if (file_hash(patients) != protocol["patients_sha256"]
            or file_hash(master) != protocol["master_sha256"]):
        raise ValueError("Annotations differ from frozen training source")
    samples, _ = load_samples(patients, master, int(protocol["fold"]), "val")
    samples = sorted(samples, key=lambda row: row["case_id"])[:limit]
    model, tokenizer = load_model(adapter_root, protocol)
    generate = make_generator(model, tokenizer, repetition_penalty, no_repeat_ngram_size)
    rows = []
    for index, sample in enumerate(samples, 1):
        result = generate_guarded(
            sample["prompt"], generate,
            fallback=lambda prompt=sample["prompt"]: render_grouped_from_prompt(prompt))
        rows.append({"case_id": sample["case_id"],
                     "input_sha256": sample["input_sha256"],
                     "target_sha256": sample["target_sha256"],
                     **result})
        print(f"Completed {index}/{len(samples)}; status={result['status']}; "
              f"attempts={len(result['attempts'])}; "
              f"reasons={[a['status'] for a in result['attempts']]}", flush=True)
    write_jsonl(output / "predictions.jsonl", rows)
    summary = {
        "split": "val", "patients": len(rows),
        "valid_on_first_attempt": sum(r["status"] == "ok" and
                                      len(r["attempts"]) == 1 for r in rows),
        "recovered_by_retry": sum(r["status"] == "ok" and
                                  len(r["attempts"]) == 2 for r in rows),
        "grouped_template_fallback": sum(r["status"] == "fallback" for r in rows),
        "rejected_after_retry": sum(r["status"] == "rejected" for r in rows),
        "retry_count": sum(len(r["attempts"]) - 1 for r in rows),
        "attempt_statuses": {status: sum(a["status"] == status for r in rows
                                 for a in r["attempts"])
                             for status in sorted({a["status"] for r in rows
                                                   for a in r["attempts"]})},
        "mean_total_seconds": sum(sum(a["seconds"] for a in r["attempts"])
                                  for r in rows) / len(rows),
        "generation": {"max_new_tokens": MAX_NEW_TOKENS,
                       "repetition_penalty": repetition_penalty,
                       "no_repeat_ngram_size": no_repeat_ngram_size},
        "protocol_sha256": file_hash(protocol_path),
        "warning": "Automatic gates cannot verify medical correctness or grouping clarity.",
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations-root", "adapter-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protocol", dest="protocol_path", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    print(run(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
