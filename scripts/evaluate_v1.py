"""Patient-level text overlap against original reports; never a clinical accuracy score."""
import argparse
import csv
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.v1_dataset import SpineVLMDataset
from src.data.report_targets import format_target
from src.io_utils import strict_loads, write_json, write_jsonl, file_hash
from src.report.experiments import wilson


def overlap(prediction, reference):
    a, b = re.findall(r"\w+", prediction.casefold()), re.findall(r"\w+", reference.casefold())
    if not a or not b: return {"word_f1": 0., "rouge_l_f1": 0.}
    # Vietnamese whitespace words are not clinical entities; label the metric accordingly.
    common = sum((Counter(a) & Counter(b)).values())
    previous = [0] * (len(b)+1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j-1]+1 if token == other else max(previous[j], current[-1]))
        previous = current
    return {"word_f1": 2*common/(len(a)+len(b)), "rouge_l_f1": 2*previous[-1]/(len(a)+len(b))}


def bootstrap_mean(values, seed=42, iterations=2000):
    if not values: return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(iterations))
    return {"n": len(values), "mean": sum(values)/len(values),
            "ci95": [draws[int(.025*iterations)], draws[min(iterations-1, int(.975*iterations))]],
            "method": "patient bootstrap percentile", "seed": seed, "iterations": iterations}


def evaluate(predictions, master, output, fold=1, split="val", protocol=None):
    if split == "test" and not protocol: raise ValueError("Lock protocol before test evaluation")
    output = Path(output)
    if output.exists(): raise ValueError("Use a new evaluation output directory")
    dataset = SpineVLMDataset(master, ".", split=split, fold=fold, require_images=False)
    references = {r["patient_id"]: r["target_text"] for r in dataset.samples}
    predictions_rows = [strict_loads(x) for x in Path(predictions).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len({r["case_id"] for r in predictions_rows}) != len(predictions_rows): raise ValueError("Duplicate patient prediction")
    rows = []
    for row in predictions_rows:
        if row["case_id"] not in references or row["split"] != split: raise ValueError("Prediction does not belong to selected split")
        text = row["prediction"]
        rows.append({"case_id": row["case_id"], **overlap(text, references[row["case_id"]]),
                     "truncated": bool(row["truncated"]), "missing_sections": not ("[MÔ TẢ]" in text and "[KẾT LUẬN]" in text)})
    metrics = {key: bootstrap_mean([r[key] for r in rows]) for key in ("word_f1", "rouge_l_f1")}
    metrics.update({key: wilson(sum(r[key] for r in rows), len(rows)) for key in ("truncated", "missing_sections")})
    metrics.update(expected_patients=len(references), evaluated_patients=len(rows),
        missing_predictions=sorted(set(references)-{r["case_id"] for r in rows}), clinical_error_rate=None,
        warning="Text overlap with original report is not grading fidelity or clinical correctness; V1 sees limited slices.")
    write_jsonl(output / "per_patient.jsonl", rows)
    write_json(output / "metrics.json", metrics)
    write_json(output / "receipt.json", {"predictions_sha256": file_hash(predictions), "master_sha256": file_hash(master),
        "fold": fold, "split": split, "protocol_sha256": file_hash(protocol) if protocol else None})
    with (output / "human_review.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, ["case_id", "clinical_error", "wrong_level", "unsupported_claim", "omission", "contradiction", "reviewer", "notes"])
        writer.writeheader()
        writer.writerows({"case_id": r["case_id"]} for r in rows)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "master", "output"): p.add_argument("--"+name, required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--protocol")
    print(evaluate(**vars(p.parse_args())))
