"""Compare an untouched Qwen3-VL with frozen V2-2 predictions on the same test set.

Only the candidate performs inference. Baseline files and all test inputs must
match the locked hashes. This is a system comparison, not a fine-tuning ablation.
"""
import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_v2_2 import aligned_records, compute_metrics, parse_report
from src.data.v2_2_dataset import load_samples
from src.io_utils import content_hash, environment_receipt, file_hash, strict_loads, write_json, write_jsonl


def read_jsonl(path):
    return [strict_loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def find_frozen_file(root, filename, expected_hash):
    matches = [p for p in sorted(Path(root).rglob(filename))
               if p.is_file() and file_hash(p) == expected_hash]
    if not matches:
        raise ValueError(f"Missing frozen {filename}; attach the original V2-2 evaluation Output")
    # Identical copies are safe; never select a different version by pathname.
    return matches[0]


def prompt_manifest(samples):
    return [{"case_id": row["case_id"], "prompt_sha256": content_hash(row["prompt"]),
             "input_sha256": row["input_sha256"], "target_sha256": row["target_sha256"]}
            for row in sorted(samples, key=lambda row: row["case_id"])]


def verify_samples(samples, baseline, config):
    if len(samples) != config["test_cases"]:
        raise ValueError("Unexpected test case count")
    if content_hash(prompt_manifest(samples)) != config["prompt_manifest_sha256"]:
        raise ValueError("Test prompts or targets differ from the locked comparison")
    return aligned_records(samples, baseline)


def summarize(records, predictions):
    times = [row["seconds"] for row in records]
    tokens = [row["new_tokens"] for row in predictions]
    return {"patients": len(records), "format_valid": sum(r["status"] == "ok" for r in records),
            "format_valid_rate": sum(r["status"] == "ok" for r in records) / len(records),
            "format_invalid_json": sum(r["status"] == "invalid_json" for r in records),
            "format_invalid_schema": sum(r["status"] == "invalid_schema" for r in records),
            "hit_token_limit": sum(r["hit_token_limit"] for r in records),
            "mean_generation_seconds": statistics.mean(times),
            "median_generation_seconds": statistics.median(times),
            "total_generation_seconds": sum(times), "mean_new_tokens": statistics.mean(tokens),
            "generated_tokens_per_second": sum(tokens) / sum(times) if sum(times) else None}


def load_candidate(config):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Run this comparison on a Kaggle CUDA GPU")
    candidate = config["candidate"]
    tokenizer = AutoTokenizer.from_pretrained(candidate["model"], revision=candidate["revision"],
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        candidate["model"], revision=candidate["revision"], trust_remote_code=False,
        torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.float16))
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_one(model, tokenizer, sample, config):
    import torch

    # Same user message as V2-2, rendered using this model's native chat template.
    chat = tokenizer.apply_chat_template([{"role": "user", "content": sample["prompt"]}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False, truncation=False)
    input_tokens = inputs["input_ids"].shape[-1]
    if input_tokens + config["generation"]["max_new_tokens"] > model.config.text_config.max_position_embeddings:
        raise ValueError("Context overflow; test inputs may not be truncated")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **config["generation"],
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    answer = outputs[0, input_tokens:]
    raw = tokenizer.decode(answer, skip_special_tokens=True)
    status, parsed = parse_report(raw)
    return {"case_id": sample["case_id"], "input_sha256": sample["input_sha256"],
            "target_sha256": sample["target_sha256"], "prompt_sha256": content_hash(sample["prompt"]),
            "rendered_prompt_sha256": content_hash(chat), "status": status, "parsed": parsed,
            "raw_output": raw, "new_tokens": len(answer), "input_tokens": input_tokens,
            "hit_token_limit": len(answer) >= config["generation"]["max_new_tokens"],
            "ended_with_eos": bool(len(answer) and answer[-1].item() == tokenizer.eos_token_id),
            "seconds": seconds}


def write_comparison(output, results, baseline_records, candidate_records):
    columns = ["model", "patients", "format_valid", "format_valid_rate", "hit_token_limit",
               "bleu4", "rouge1_f1", "rougeL_f1", "bertscore_f1", "mean_generation_seconds",
               "median_generation_seconds", "mean_new_tokens", "generated_tokens_per_second"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, metrics in results.items():
            row = {field: metrics[field] for field in columns if field != "model"}
            row.update(model=name, bleu4=metrics["bleu4"]["score"])
            writer.writerow(row)
    review_fields = ["case_id", "reference_findings", "reference_impression"]
    for prefix in ("finetuned", "qwen3"):
        review_fields += [prefix + "_" + field for field in
                          ("status", "findings", "impression", "raw_output", "seconds")]
    review_fields += ["preferred", "unsupported_claim", "wrong_level", "omission", "contradiction", "reviewer", "notes"]
    baseline_by_id = {row["case_id"]: row for row in baseline_records}
    with (output / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for candidate in candidate_records:
            original = baseline_by_id[candidate["case_id"]]
            row = {"case_id": candidate["case_id"],
                   "reference_findings": candidate["reference"]["findings"],
                   "reference_impression": candidate["reference"]["impression"]}
            for prefix, record in (("finetuned", original), ("qwen3", candidate)):
                prediction = record["prediction"] or {}
                row.update({prefix + "_status": record["status"],
                            prefix + "_findings": prediction.get("findings", ""),
                            prefix + "_impression": prediction.get("impression", ""),
                            prefix + "_raw_output": record["raw_output"],
                            prefix + "_seconds": record["seconds"]})
            writer.writerow(row)


def run(input_root, config_path, output):
    config = strict_loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    paths = {key: find_frozen_file(input_root, spec["filename"], spec["sha256"])
             for key, spec in config["frozen_files"].items()}
    samples, counts = load_samples(paths["patients"], paths["master"], config["fold"], "test")
    samples = sorted(samples, key=lambda row: row["case_id"])
    baseline_predictions = read_jsonl(paths["baseline_predictions"])
    baseline_records = verify_samples(samples, baseline_predictions, config)
    baseline_receipt = strict_loads(paths["baseline_receipt"].read_text(encoding="utf-8"))
    if baseline_receipt["protocol_sha256"] != config["baseline_protocol_sha256"]:
        raise ValueError("Frozen baseline protocol differs")
    output.mkdir(parents=True)
    write_jsonl(output / "inputs.jsonl", samples)
    write_jsonl(output / "prompt_manifest.jsonl", prompt_manifest(samples))
    write_json(output / "baseline_original_metrics.json",
               strict_loads(paths["baseline_metrics"].read_text(encoding="utf-8")))
    receipt = {"comparison_config_sha256": file_hash(config_path), "config": config,
               "source_paths": {k: str(v) for k, v in paths.items()}, "eligible_counts": counts,
               "baseline_environment": baseline_receipt["environment"],
               "candidate_environment": environment_receipt(),
               "limitations": ["Different base models: this does not isolate the effect of fine-tuning.",
                               "Identical user prompts; each model uses its own chat template/tokenizer.",
                               "Baseline inference timings are historical, not a same-session speed benchmark.",
                               "Fold-1 test has been examined previously; this is exploratory.",
                               "Both arms are rescored with the same metric implementation and runtime.",
                               "JSON validity and text overlap are not clinical correctness."]}
    write_json(output / "receipt.json", receipt)
    print(f"Verified {len(samples)} frozen test cases and all cached baseline files", flush=True)
    import torch
    from transformers import set_seed
    set_seed(42)
    loading_start = time.monotonic()
    model, tokenizer = load_candidate(config)
    receipt.update(model_load_seconds=time.monotonic() - loading_start,
                   gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                   candidate_generation_defaults=model.generation_config.to_dict())
    write_json(output / "receipt.json", receipt)
    predictions = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(samples, 1):
            row = generate_one(model, tokenizer, sample, config)
            predictions.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            elapsed = sum(p["seconds"] for p in predictions)
            eta = elapsed / index * (len(samples) - index)
            write_json(output / "progress.json", {"completed": index, "total": len(samples),
                       "generation_seconds": elapsed, "estimated_remaining_generation_seconds": eta,
                       "format_valid": sum(p["status"] == "ok" for p in predictions)})
            print(f"Completed {index}/{len(samples)}; status={row['status']}; tokens={row['new_tokens']}; "
                  f"seconds={row['seconds']:.1f}; remaining_generation_minutes={eta/60:.1f}", flush=True)
    candidate_records = aligned_records(samples, predictions)
    write_jsonl(output / "per_patient.jsonl", candidate_records)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    results = {}
    metric_start = time.monotonic()
    for name, records, raw in (("qwen2_5_3b_finetuned_cached", baseline_records, baseline_predictions),
                               ("qwen3_vl_4b_instruct_untuned", candidate_records, predictions)):
        print(f"Scoring {name} on all {len(records)} cases", flush=True)
        metrics = summarize(records, raw)
        metrics.update(compute_metrics(records, model_name=config["bertscore_model"], device="cuda"))
        results[name] = metrics
        write_json(output / (name + "_metrics.json"), metrics)
        gc.collect()
        torch.cuda.empty_cache()
    receipt["metric_seconds"] = time.monotonic() - metric_start
    write_json(output / "receipt.json", receipt)
    write_json(output / "comparison.json", {"results": results, "limitations": receipt["limitations"]})
    write_comparison(output, results, baseline_records, candidate_records)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--config", dest="config_path", default="configs/v2_2_qwen3_comparison.json")
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))
