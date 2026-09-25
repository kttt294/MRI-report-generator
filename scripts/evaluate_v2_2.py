"""Generate V2-2 reports and score them against original doctor-authored text.

The frozen test protocol uses exactly one greedy generation per case. Invalid
JSON is counted as a failure; no retry, fallback, or post-hoc text repair.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Set before torch/transformers imports: Kaggle T4 x2 otherwise replicates a
# quantized model across both GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cloud_prepare import prepare
from src.data.v2_2_dataset import load_samples
from src.io_utils import environment_receipt, file_hash, strict_loads, write_json, write_jsonl


def parse_report(text):
    try:
        value = strict_loads(text.strip())
    except (TypeError, ValueError):
        return "invalid_json", None
    if (not isinstance(value, dict) or set(value) != {"findings", "impression"}
            or any(not isinstance(value[k], str) or not value[k].strip()
                   for k in ("findings", "impression"))):
        return "invalid_schema", None
    return "ok", {k: value[k].strip() for k in ("findings", "impression")}


def join_sections(value):
    return f"Findings:\n{value['findings']}\n\nImpression:\n{value['impression']}"


def aligned_records(samples, predictions):
    refs = {row["case_id"]: row for row in samples}
    preds = {row["case_id"]: row for row in predictions}
    if len(refs) != len(samples) or len(preds) != len(predictions) or set(refs) != set(preds):
        raise ValueError("Duplicate or unmatched test cases")
    records = []
    for case_id in sorted(refs):
        sample, pred = refs[case_id], preds[case_id]
        if (pred["input_sha256"] != sample["input_sha256"]
                or pred["target_sha256"] != sample["target_sha256"]):
            raise ValueError(f"Prediction is bound to different input/target: {case_id}")
        status, parsed = parse_report(pred["raw_output"])
        if status != pred["status"] or parsed != pred["parsed"]:
            raise ValueError(f"Prediction status/JSON mismatch: {case_id}")
        reference = strict_loads(sample["completion"])
        records.append({"case_id": case_id, "status": status,
                        "reference": reference, "prediction": parsed,
                        "raw_output": pred["raw_output"],
                        "hit_token_limit": pred["hit_token_limit"],
                        "seconds": pred["seconds"]})
    return records


def compute_metrics(records, model_name="bert-base-multilingual-cased", device="cuda", batch_size=8):
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score

    references = [join_sections(row["reference"]) for row in records]
    predictions = [join_sections(row["prediction"]) if row["status"] == "ok" else ""
                   for row in records]
    valid_indices = [i for i, row in enumerate(records) if row["status"] == "ok"]
    bleu = BLEU(tokenize="none", smooth_method="exp", effective_order=False).corpus_score(
        predictions, [references])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rouge = {key: [] for key in ("rouge1", "rougeL")}
    for reference, prediction in zip(references, predictions):
        scores = scorer.score(reference, prediction)
        for key in rouge:
            rouge[key].append(scores[key].fmeasure)
    bert_f1 = [0.0] * len(records)
    if valid_indices:
        _, _, f1 = bert_score([predictions[i] for i in valid_indices],
                              [references[i] for i in valid_indices], model_type=model_name,
                              lang="vi", device=device, batch_size=batch_size,
                              rescale_with_baseline=False, verbose=False)
        for index, score in zip(valid_indices, f1.tolist()):
            bert_f1[index] = float(score)
    return {
        "bleu4": {"score": bleu.score, "tokenization": "none", "smoothing": "exp"},
        "rouge1_f1": sum(rouge["rouge1"]) / len(records),
        "rougeL_f1": sum(rouge["rougeL"]) / len(records),
        "bertscore_f1": sum(bert_f1) / len(records),
        "bertscore_model": model_name,
        "invalid_prediction_policy": "empty prediction for BLEU/ROUGE and zero for BERTScore",
    }


def load_model(adapter_root, protocol):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from src.models.runtime import choose_dtype

    if not torch.cuda.is_available():
        raise RuntimeError("V2-2 evaluation requires a Kaggle GPU")
    dtype = choose_dtype("auto")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_root), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        protocol["base_model"], revision=protocol["base_revision"],
        torch_dtype=dtype, device_map={"": 0}, trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_quant_type="nf4",
                                                bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(base, str(adapter_root), is_trainable=False)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt, max_new_tokens):
    import torch
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    seconds = time.monotonic() - start
    token_count = output.shape[-1] - encoded["input_ids"].shape[-1]
    raw = tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:],
                           skip_special_tokens=True)
    return raw, token_count, seconds


def run(annotations_root, adapter_root, protocol_path, output, split="test", limit=None,
        skip_metrics=False):
    if split not in {"val", "test"} or (split == "test" and (limit or skip_metrics)):
        raise ValueError("The test run must score all cases; use val for a technical smoke test")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    protocol = strict_loads(Path(protocol_path).read_text(encoding="utf-8"))
    adapter_root = Path(adapter_root).resolve()
    adapter_file = adapter_root / "adapter_model.safetensors"
    if file_hash(adapter_file) != protocol["adapter_sha256"]:
        raise ValueError("Adapter SHA-256 differs from the frozen protocol")
    output.mkdir(parents=True)
    prepared = prepare(annotations_root, output / "derived",
                       source_versions={"annotations": protocol["annotations_dataset"]})
    patients = output / "derived/dataset/dataset_patients.jsonl"
    master = output / "derived/dataset/dataset_master.csv"
    if (file_hash(patients) != protocol["patients_sha256"]
            or file_hash(master) != protocol["master_sha256"]):
        raise ValueError("Derived annotations differ from the frozen training source")
    samples, counts = load_samples(patients, master, int(protocol["fold"]), split)
    if split == "test" and len(samples) != protocol["test_cases_with_both_sections"]:
        raise ValueError("Unexpected number of eligible test cases")
    samples = sorted(samples, key=lambda row: row["case_id"])
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    write_json(output / "receipt.json", {
        "protocol_sha256": file_hash(protocol_path), "adapter_sha256": file_hash(adapter_file),
        "patients_sha256": file_hash(patients), "master_sha256": file_hash(master),
        "split": split, "fold": protocol["fold"], "cases": len(samples),
        "missing_sections": counts, "prepared": prepared, "environment": environment_receipt()})
    model, tokenizer = load_model(adapter_root, protocol)
    predictions = []
    path = output / "predictions.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as writer:
        for index, sample in enumerate(samples, 1):
            raw, token_count, seconds = generate_one(
                model, tokenizer, sample["prompt"], protocol["generation"]["max_new_tokens"])
            status, parsed = parse_report(raw)
            row = {"case_id": sample["case_id"], "input_sha256": sample["input_sha256"],
                   "target_sha256": sample["target_sha256"], "status": status,
                   "parsed": parsed, "raw_output": raw, "new_tokens": token_count,
                   "hit_token_limit": token_count >= protocol["generation"]["max_new_tokens"],
                   "seconds": seconds}
            writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            writer.flush()
            predictions.append(row)
            print(f"Completed {index}/{len(samples)}; status={status}; seconds={seconds:.1f}", flush=True)
    records = aligned_records(samples, predictions)
    write_jsonl(output / "per_patient.jsonl", records)
    with (output / "human_review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["case_id", "format_status", "reference_findings", "reference_impression",
                  "prediction_findings", "prediction_impression", "unsupported_claim",
                  "wrong_level", "omission", "contradiction", "reviewer", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            prediction = row["prediction"] or {}
            writer.writerow({"case_id": row["case_id"], "format_status": row["status"],
                             "reference_findings": row["reference"]["findings"],
                             "reference_impression": row["reference"]["impression"],
                             "prediction_findings": prediction.get("findings", ""),
                             "prediction_impression": prediction.get("impression", "")})
    del model
    import torch
    torch.cuda.empty_cache()
    result = {"fold": protocol["fold"], "split": split, "patients": len(records),
              "format_valid": sum(row["status"] == "ok" for row in records),
              "format_valid_rate": sum(row["status"] == "ok" for row in records) / len(records),
              "format_invalid_json": sum(row["status"] == "invalid_json" for row in records),
              "format_invalid_schema": sum(row["status"] == "invalid_schema" for row in records),
              "hit_token_limit": sum(row["hit_token_limit"] for row in records),
              "mean_generation_seconds": sum(row["seconds"] for row in records) / len(records),
              "protocol_sha256": file_hash(protocol_path),
              "warning": "Text overlap and JSON validity do not establish clinical correctness."}
    if not skip_metrics:
        result.update(compute_metrics(records, model_name=protocol["metrics"]["bertscore_model"]))
    write_json(output / "metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations-root", "adapter-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protocol", dest="protocol_path", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-metrics", action="store_true")
    print(run(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
