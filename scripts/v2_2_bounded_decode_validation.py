"""Validation-only pilot of bounded prompting plus repetition controls for V2-2.

This is a decoding experiment, not a replacement for the frozen fold-1 test.
The prior prompt-only retry used the same prompt and 1024-token cap on one
failing validation case, allowing a direct diagnostic comparison there.
"""
import argparse
import os
import re
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
from src.report.v2_2_quality import inspect_report, retry_prompt


MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 6


def repeated_fivegram_fraction(text):
    words = re.findall(r"\S+", text.lower())
    grams = [tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def generate_one(model, tokenizer, prompt):
    import torch

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True)
    encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    seconds = time.monotonic() - start
    tokens = output.shape[-1] - encoded["input_ids"].shape[-1]
    raw = tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:],
                           skip_special_tokens=True)
    return raw, tokens, seconds


def run(annotations_root, adapter_root, protocol_path, output, limit=2):
    if limit < 1:
        raise ValueError("limit must be positive")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    protocol = strict_loads(Path(protocol_path).read_text(encoding="utf-8"))
    adapter_root = Path(adapter_root).resolve()
    if file_hash(adapter_root / "adapter_model.safetensors") != protocol["adapter_sha256"]:
        raise ValueError("Adapter differs from the frozen baseline protocol")
    output.mkdir(parents=True)
    prepare(annotations_root, output / "derived",
            source_versions={"annotations": protocol["annotations_dataset"]})
    patients = output / "derived/dataset/dataset_patients.jsonl"
    master = output / "derived/dataset/dataset_master.csv"
    if (file_hash(patients) != protocol["patients_sha256"]
            or file_hash(master) != protocol["master_sha256"]):
        raise ValueError("Annotations differ from the baseline source")
    samples, _ = load_samples(patients, master, int(protocol["fold"]), "val")
    samples = sorted(samples, key=lambda row: row["case_id"])[:limit]
    model, tokenizer = load_model(adapter_root, protocol)
    rows = []
    for index, sample in enumerate(samples, 1):
        prompt = retry_prompt(sample["prompt"], "invalid_json")
        raw, tokens, seconds = generate_one(model, tokenizer, prompt)
        gate = inspect_report(raw)
        rows.append({
            "case_id": sample["case_id"],
            "input_sha256": sample["input_sha256"],
            "status": gate["status"], "tokens": tokens,
            "hit_token_limit": tokens >= MAX_NEW_TOKENS,
            "seconds": seconds, "chars": len(raw),
            "repeated_fivegram_fraction": repeated_fivegram_fraction(raw),
            "raw_output": raw,
        })
        print(f"Completed {index}/{len(samples)}; status={gate['status']}; "
              f"tokens={tokens}; seconds={seconds:.1f}", flush=True)
    write_jsonl(output / "predictions.jsonl", rows)
    summary = {
        "split": "val", "patients": len(rows),
        "format_and_length_valid": sum(row["status"] == "ok" for row in rows),
        "hit_token_limit": sum(row["hit_token_limit"] for row in rows),
        "mean_repeated_fivegram_fraction":
            sum(row["repeated_fivegram_fraction"] for row in rows) / len(rows),
        "mean_generation_seconds": sum(row["seconds"] for row in rows) / len(rows),
        "generation": {"max_new_tokens": MAX_NEW_TOKENS,
                       "repetition_penalty": REPETITION_PENALTY,
                       "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE},
        "protocol_sha256": file_hash(protocol_path),
        "warning": "Format and repetition checks do not establish clinical correctness.",
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations-root", "adapter-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protocol", dest="protocol_path", required=True)
    parser.add_argument("--limit", type=int, default=2)
    print(run(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
