from scripts.evaluate_v2_nlp import aligned_pairs, join_sections, load_references
from src.contracts.report_input import ReportRequest
from src.io_utils import content_hash, write_jsonl
from src.report.planner import make_plan, template_draft


def test_v2_nlp_targets_are_split_bound_and_predictions_align(tmp_path, request_data):
    request = ReportRequest.model_validate(request_data)
    target = template_draft(make_plan(request)).model_dump()
    reviewed = {
        "case_id": request_data["case_id"], "request": request_data,
        "input_sha256": content_hash(request_data),
        "folds": {f"fold{i}": "test" for i in range(1, 6)},
        "review_status": "accepted", "reviewed_by": "reviewer", "target_kind": "human_reviewed_scoped",
        "target": target,
    }
    path = tmp_path / "reviewed.jsonl"
    write_jsonl(path, [reviewed])
    references = load_references(path, fold=1, split="test")
    findings, impression = join_sections("F", "I").split("\n\n", 1)
    predictions = {request_data["case_id"]: {"case_id": request_data["case_id"], "result": {
        "findings": findings.removeprefix("Findings:\n"), "impression": impression.removeprefix("Impression:\n"),
        "generation_method": "llm", "status": "ok"}}}
    pairs = aligned_pairs(references, predictions)
    assert len(pairs) == 1
    assert pairs[0]["reference_is_exact_template"] is True
