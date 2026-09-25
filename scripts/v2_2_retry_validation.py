"""Technical validation of one bounded V2-2 retry after a failed quality gate.

Only fold-1 validation cases are allowed here; this script does not tune on
test examples and does not claim clinical correctness.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cloud_prepare import prepare
from scripts.evaluate_v2_2 import generate_one, load_model
from src.data.v2_2_dataset import load_samples
from src.io_utils import file_hash, strict_loads, write_json, write_jsonl
from src.report.v2_2_quality import inspect_report, retry_prompt


RETRY_MAX_NEW_TOKENS = 1024


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
        first_raw, first_tokens, first_seconds = generate_one(
            model, tokenizer, sample["prompt"], protocol["generation"]["max_new_tokens"])
        first_gate = inspect_report(first_raw)
        attempts = [{"raw_output": first_raw, "tokens": first_tokens,
                     "seconds": first_seconds, "status": first_gate["status"]}]
        if first_gate["status"] != "ok":
            revised = retry_prompt(sample["prompt"], first_gate["status"])
            raw, tokens, seconds = generate_one(model, tokenizer, revised, RETRY_MAX_NEW_TOKENS)
            second_gate = inspect_report(raw)
            attempts.append({"raw_output": raw, "tokens": tokens, "seconds": seconds,
                             "status": second_gate["status"]})
        rows.append({"case_id": sample["case_id"], "input_sha256": sample["input_sha256"],
                     "attempts": attempts, "final_status": attempts[-1]["status"],
                     "accepted_attempt": 1 if first_gate["status"] == "ok" else
                     2 if attempts[-1]["status"] == "ok" else None})
        print(f"Completed {index}/{len(samples)}; first={attempts[0]['status']}; "
              f"final={attempts[-1]['status']}; attempts={len(attempts)}", flush=True)
    write_jsonl(output / "attempts.jsonl", rows)
    summary = {"split": "val", "patients": len(rows),
               "first_pass_accepted": sum(row["accepted_attempt"] == 1 for row in rows),
               "retry_accepted": sum(row["accepted_attempt"] == 2 for row in rows),
               "final_rejected": sum(row["accepted_attempt"] is None for row in rows),
               "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
               "protocol_sha256": file_hash(protocol_path),
               "warning": "This quality gate checks format and length only, not clinical correctness."}
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
