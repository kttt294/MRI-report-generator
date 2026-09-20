"""Evaluate delivered V2 reports against reviewed, split-bound text targets.

This is a text-fidelity evaluation, not a measure of MRI diagnostic accuracy:
the V2 report engine receives the structured grading JSON as its input.
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts.report_input import ReportRequest
from src.contracts.report_output import ReportDraft
from src.io_utils import file_hash, strict_loads, write_json, write_jsonl
from src.report.planner import make_plan, template_draft
from src.report.validate import render


def render_target(row):
    request = ReportRequest.model_validate(row["request"])
    target = ReportDraft.model_validate(row["target"])
    findings, impression = render(target)
    template = template_draft(make_plan(request)).model_dump()
    return {
        "case_id": row["case_id"],
        "text": join_sections(findings, impression),
        "is_exact_template": target.model_dump() == template,
    }


def join_sections(findings, impression):
    if not isinstance(findings, str) or not findings.strip():
        raise ValueError("findings must be a non-empty string")
    if not isinstance(impression, str) or not impression.strip():
        raise ValueError("impression must be a non-empty string")
    return f"Findings:\n{findings.strip()}\n\nImpression:\n{impression.strip()}"


def load_references(path, fold, split):
    rows, seen = {}, set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = strict_loads(line)
        case_id = row["case_id"]
        if case_id in seen:
            raise ValueError(f"Duplicate reviewed target: {case_id}")
        seen.add(case_id)
        if row.get("review_status") != "accepted" or not row.get("reviewed_by") or row.get("target_kind") != "human_reviewed_scoped":
            raise ValueError("NLP references must be explicitly human-reviewed scoped targets")
        if row["folds"].get(f"fold{fold}") == split:
            rows[case_id] = render_target(row)
    if not rows:
        raise ValueError(f"No reviewed references for fold {fold} {split}")
    return rows


def prediction_text(row):
    result = row.get("result", row)
    return join_sections(result.get("findings"), result.get("impression"))


def load_predictions(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = strict_loads(line)
        case_id = row["case_id"]
        if case_id in rows:
            raise ValueError(f"Duplicate prediction: {case_id}")
        rows[case_id] = row
    if not rows:
        raise ValueError("No predictions")
    return rows


def aligned_pairs(references, predictions):
    missing = sorted(set(references) - set(predictions))
    unexpected = sorted(set(predictions) - set(references))
    if missing or unexpected:
        raise ValueError(f"Predictions and references differ; missing={missing}, unexpected={unexpected}")
    pairs = []
    for case_id in sorted(references):
        prediction = predictions[case_id]
        result = prediction.get("result", prediction)
        pairs.append({
            "case_id": case_id,
            "reference": references[case_id]["text"],
            "prediction": prediction_text(prediction),
            "reference_is_exact_template": references[case_id]["is_exact_template"],
            "generation_method": result.get("generation_method"),
            "status": result.get("status"),
        })
    return pairs


def average(scores, field):
    return sum(getattr(score, field) for score in scores) / len(scores)


def compute_metrics(pairs, bertscore_model, device, batch_size):
    try:
        from sacrebleu.metrics import BLEU
        from rouge_score import rouge_scorer
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError("Install requirements-evaluation.txt before running this evaluator") from exc

    predictions = [row["prediction"] for row in pairs]
    references = [row["reference"] for row in pairs]
    bleu = BLEU(tokenize="none", smooth_method="exp", effective_order=False).corpus_score(predictions, [references])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rouge = {name: [scorer.score(reference, prediction)[name] for reference, prediction in zip(references, predictions)]
             for name in ("rouge1", "rougeL")}
    precision, recall, f1 = bert_score(predictions, references, model_type=bertscore_model,
                                        lang="vi", device=device, batch_size=batch_size,
                                        rescale_with_baseline=False, verbose=True)
    return {
        "bleu4": {"score": bleu.score, "tokenization": "none", "smoothing": "exp"},
        "rouge1": {field: average(rouge["rouge1"], field) for field in ("precision", "recall", "fmeasure")},
        "rougeL": {field: average(rouge["rougeL"], field) for field in ("precision", "recall", "fmeasure")},
        "bertscore": {
            "model": bertscore_model,
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
            "rescale_with_baseline": False,
        },
    }


def evaluate(predictions, reviewed_targets, output, fold=1, split="val", protocol=None,
             bertscore_model="bert-base-multilingual-cased", device=None, batch_size=8):
    if split == "test" and not protocol:
        raise ValueError("Lock a protocol file before final test NLP evaluation")
    if fold not in range(1, 6) or split not in {"train", "val", "test"}:
        raise ValueError("Invalid fold or split")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    references = load_references(reviewed_targets, fold, split)
    pairs = aligned_pairs(references, load_predictions(predictions))
    output.mkdir(parents=True)
    metrics = compute_metrics(pairs, bertscore_model, device, batch_size)
    template_references = sum(row["reference_is_exact_template"] for row in pairs)
    methods = collections.Counter(row["generation_method"] for row in pairs)
    metrics.update({
        "patients": len(pairs),
        "fold": fold,
        "split": split,
        "generation_methods": dict(methods),
        "reference_exact_template": {"count": template_references, "rate": template_references / len(pairs)},
        "warning": (
            "These are controlled-text fidelity metrics. A reference that equals the deterministic template "
            "cannot establish clinical quality or a LoRA advantage over that template."
        ),
        "inputs": {
            "predictions_sha256": file_hash(predictions),
            "reviewed_targets_sha256": file_hash(reviewed_targets),
            "protocol_sha256": file_hash(protocol) if protocol else None,
        },
    })
    write_jsonl(output / "per_patient.jsonl", pairs)
    write_json(output / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reviewed-targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--protocol")
    parser.add_argument("--bertscore-model", default="bert-base-multilingual-cased")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(evaluate(**vars(args)))


if __name__ == "__main__":
    main()
