"""Export requests separately from targets and folds; never modify the sources."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.v2_adapter import DatasetAdapter
from src.data.report_targets import vi_sections
from src.io_utils import strict_loads, file_hash, write_json, write_jsonl


def build(patients_path, master_csv, output):
    output = Path(output)
    if output.resolve() in {Path(patients_path).resolve().parent, Path(master_csv).resolve().parent}:
        raise ValueError("Choose a separate derived output directory")
    adapter = DatasetAdapter(master_csv)
    requests, targets, manifest, errors, seen = [], [], [], [], set()
    for line_no, line in enumerate(Path(patients_path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            patient = strict_loads(line)
            pid = str(patient["patient_id"])
            if pid in seen:
                raise ValueError("Duplicate patient ID")
            seen.add(pid)
            request, corrections = adapter.convert(patient)
            findings, impression = vi_sections(patient.get("reports", {}).get("vi", {}))
            if set(patient["folds"]) != {f"fold{i}" for i in range(1, 6)}:
                raise ValueError("Exactly five folds are required")
            for fold, split in patient["folds"].items():
                expected = {adapter.rows[(pid, x.level)][f"{fold}_split"] for x in request.levels}
                if expected != {split} or split not in {"train", "val", "test"}:
                    raise ValueError(f"Inconsistent {fold} split")
            requests.append(request.model_dump())
            targets.append({"case_id": pid, "findings": findings, "impression": impression,
                            "review_status": "unreviewed", "scope": "original_full_report"})
            manifest.append({"case_id": pid, "folds": patient["folds"], "corrections": corrections,
                             "has_both_sections": bool(findings and impression)})
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"line": line_no, "error": str(exc)})
    write_jsonl(output / "requests.jsonl", requests)
    write_jsonl(output / "references_unreviewed.jsonl", targets)
    write_jsonl(output / "manifest.jsonl", manifest)
    receipt = {"adapter": "legacy-adapter/1.0", "patients_sha256": file_hash(patients_path),
               "master_sha256": adapter.source_sha256, "requests": len(requests), "errors": errors,
               "missing_cells_recovered": sum(len(m["corrections"]) for m in manifest),
               "input_kind": "annotations_not_vision_predictions"}
    write_json(output / "build_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", default="dataset/dataset_patients.jsonl")
    parser.add_argument("--master", default="dataset/dataset_master.csv")
    parser.add_argument("--output", default="output/v2_data")
    args = parser.parse_args()
    receipt = build(args.patients, args.master, args.output)
    print({key: receipt[key] for key in ("requests", "missing_cells_recovered")})
    raise SystemExit(1 if receipt["errors"] else 0)
