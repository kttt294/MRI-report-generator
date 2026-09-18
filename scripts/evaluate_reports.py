"""Run one predeclared ablation on a patient split; export blinded human review sheet."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.contracts.report_input import ReportRequest
from src.io_utils import strict_loads, write_json, write_jsonl, file_hash, environment_receipt
from src.report.experiments import ablate, summarize


def run(requests, manifest, output, arm="R0", fold=1, split="val", config=None, protocol=None, matched_v1_cohort=False):
    if split == "test" and not protocol: raise ValueError("Lock a protocol file before final test evaluation")
    if fold not in range(1, 6) or split not in {"train", "val", "test"}: raise ValueError("Invalid split/fold")
    output = Path(output)
    if output.exists(): raise ValueError("Use a new evaluation output directory")
    entries = [strict_loads(x) for x in Path(manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len({e["case_id"] for e in entries}) != len(entries): raise ValueError("Duplicate manifest case")
    selected = {e["case_id"] for e in entries if e["folds"][f"fold{fold}"] == split
                and (not matched_v1_cohort or e["has_both_sections"])}
    if not selected: raise ValueError("Empty evaluation split")
    backend = None
    if arm != "R0":
        from src.report.backends.hf_local import HFLocalBackend
        if not config: raise ValueError("LLM arm requires --config")
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        if arm == "R5" and not cfg.get("adapter_path"): raise ValueError("R5 requires a reviewed-target adapter")
        if arm != "R5" and cfg.get("adapter_path"): raise ValueError("Use R5 for adapters; R1–R4 use same base model")
        backend = HFLocalBackend(**{k: cfg[k] for k in ("model_path", "revision", "max_new_tokens", "max_input_tokens",
            "max_time_seconds", "use_4bit", "local_files_only", "adapter_path") if k in cfg})
    rows, seen = [], set()
    for line in Path(requests).read_text(encoding="utf-8").splitlines():
        req = ReportRequest.model_validate(strict_loads(line))
        if req.case_id in seen: raise ValueError("Duplicate request case")
        seen.add(req.case_id)
        if req.case_id in selected: rows.append(ablate(req, arm, backend))
    if selected - seen: raise ValueError("Manifest references missing requests")
    write_jsonl(output / "predictions.jsonl", rows)
    write_json(output / "metrics.json", summarize(rows))
    write_json(output / "receipt.json", {"arm": arm, "fold": fold, "split": split, "repairs": 0,
        "matched_v1_cohort": matched_v1_cohort,
        "requests_sha256": file_hash(requests), "manifest_sha256": file_hash(manifest),
        "protocol_sha256": file_hash(protocol) if protocol else None,
        "config_sha256": file_hash(config) if config else None, "environment": environment_receipt()})
    with (output / "human_review.csv").open("w", encoding="utf-8", newline="") as f:
        columns = ["case_id", "clinical_error", "wrong_level", "unsupported_claim", "omission", "contradiction", "reviewer", "notes"]
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows({"case_id": r["case_id"]} for r in rows)
    return summarize(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("requests", "manifest", "output"): p.add_argument("--" + name, required=True)
    p.add_argument("--arm", choices=[f"R{i}" for i in range(6)], default="R0")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--config")
    p.add_argument("--protocol")
    p.add_argument("--matched-v1-cohort", action="store_true")
    print(run(**vars(p.parse_args())))
