"""Summarize completed patient-level human review; never convert missing review to zero."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.report.experiments import wilson
from src.io_utils import write_json, file_hash


def summarize(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f: rows = list(csv.DictReader(f))
    if len({r["case_id"] for r in rows}) != len(rows): raise ValueError("Duplicate patient; adjudicate multiple reviewers first")
    metrics = {}
    for field in ("clinical_error", "wrong_level", "unsupported_claim", "omission", "contradiction"):
        values = []
        for row in rows:
            value = row.get(field, "").strip()
            if value:
                if value not in {"0", "1"} or not row.get("reviewer", "").strip():
                    raise ValueError("Review values must be 0/1 with reviewer attribution")
                values.append(int(value))
        metrics[field] = {**wilson(sum(values), len(values)), "unreviewed": len(rows)-len(values)}
    return {"source_sha256": file_hash(path), "patients": len(rows), "metrics": metrics,
            "method": "Wilson 95% intervals over patients; descriptive on reviewed subset, not blinded-study proof"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    write_json(a.output, summarize(a.input))
