import ast
import csv
from pathlib import Path
import nbformat
import pytest
from src.io_utils import write_json, write_jsonl, strict_loads, file_hash, content_hash
from src.contracts.report_input import LEVELS
from src.report.planner import template_draft, make_plan
from src.report.experiments import ablate, summarize, wilson
from src.report.backends.base import Completion
from src.data.v2_dataset import load_reviewed_targets
from scripts.cloud_prepare import prepare
from scripts.make_cloud_notebooks import notebook
from scripts.summarize_review import summarize as summarize_review


def test_notebooks_thin_clear_and_valid():
    for task in ("v1-train", "v2-generate", "v2-train"):
        nb = notebook(task)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.outputs == [] and cell.execution_count is None
                tree = ast.parse(cell.source)
                assert not any(isinstance(x, (ast.ClassDef, ast.FunctionDef)) for x in ast.walk(tree))


def test_review_gate_and_hash_binding(report_request, tmp_path):
    row = {"case_id": report_request.case_id, "request": report_request.model_dump(),
           "input_sha256": content_hash(report_request.model_dump()), "review_status": "pending",
           "reviewed_by": "", "target_kind": "template_draft_not_clinical_ground_truth",
           "folds": {f"fold{i}": "train" for i in range(1, 6)},
           "target": template_draft(make_plan(report_request)).model_dump()}
    path = tmp_path / "targets.jsonl"
    write_jsonl(path, [row])
    with pytest.raises(ValueError): load_reviewed_targets(path, 1, "train")
    row.update(review_status="accepted", reviewed_by="synthetic-reviewer", target_kind="human_reviewed_scoped")
    write_jsonl(path, [row])
    assert len(load_reviewed_targets(path, 1, "train")) == 1
    row["target"]["findings"][0]["text"] = "invented clinical finding"
    write_jsonl(path, [row])
    with pytest.raises(ValueError): load_reviewed_targets(path, 1, "train")
    row["input_sha256"] = "b" * 64
    write_jsonl(path, [row])
    with pytest.raises(ValueError): load_reviewed_targets(path, 1, "train")


def test_metrics_do_not_invent_clinical_rate(report_request):
    result = summarize([ablate(report_request, "R0")])
    assert result["clinical_error_rate"] is None
    assert result["format_failure"]["n"] == 1
    assert 0 < wilson(0, 1)["ci95"][1] < 1
    assert wilson(0, 0)["rate"] is None
    with pytest.raises(ValueError): summarize([ablate(report_request, "R0")] * 2)


def test_prompt_and_structured_arms_use_same_prompt(report_request):
    class Backend:
        model_id = "synthetic"
        calls = []
        def generate(self, prompt, schema=None):
            self.calls.append((prompt, schema))
            return Completion('{"findings":"synthetic findings","impression":"synthetic impression"}')
    backend = Backend()
    for arm in ("R1", "R2"):
        row = ablate(report_request, arm, backend)
        assert row["format_valid"] and row["catalog_conformant"] is None
    assert backend.calls[0][0] == backend.calls[1][0]
    assert backend.calls[0][1] is None and backend.calls[1][1] is not None
    assert report_request.case_id not in backend.calls[0][0]


def test_two_stage_trace_and_validator(report_request):
    draft = template_draft(make_plan(report_request)).model_dump()
    class Backend:
        model_id = "synthetic"
        calls = 0
        def generate(self, prompt):
            section = ["findings", "impression"][self.calls]
            self.calls += 1
            from src.io_utils import canonical_json
            return Completion(canonical_json({section: draft[section]}))
    row = ablate(report_request, "R4", Backend())
    assert len(row["raw_calls"]) == 2 and row["catalog_conformant"] and not row["fallback"]


def test_unreviewed_not_counted_as_zero(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("case_id,clinical_error,reviewer\nsynthetic-a,,\nsynthetic-b,1,synthetic-reviewer\n", encoding="utf-8")
    result = summarize_review(path)
    assert result["metrics"]["clinical_error"]["n"] == 1
    assert result["metrics"]["clinical_error"]["unreviewed"] == 1


def test_raw_etl_missing_preserved_source_untouched(tmp_path):
    source, output = tmp_path / "input", tmp_path / "derived"
    def csv_file(relative, rows):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    rows = [{"Patient ID": "001", "level": level, "IVD label": i+1, "month": "01",
        "Modic": 0, "UP endplate": 0, "LOW endplate": 0, "Spondylolisthesis": 0,
        "Disc herniation": 0, "Disc narrowing": 0, "Disc bulging": "" if i == 3 else 0,
        "Pfirrman grade": 1} for i, level in enumerate(LEVELS)]
    csv_file("grading/grading_all.csv", rows)
    csv_file("localize/disc_localization.csv", [{"patient_id": "001", "level": level, "in_cohort": 1,
        "i": 1, "j": 1, "k": 1, "x_lps": 1, "y_lps": 1, "z_lps": 1, "volume": "synthetic.nii.gz",
        "spacing_i": 1, "spacing_j": 1, "spacing_k": 1, "source": "synthetic", "qc_status": "test"} for level in LEVELS])
    for i in range(1, 6): csv_file(f"folds/fold{i}/train.csv", [{"Patient ID": "001"}])
    write_json(source / "reports_json/report/001.json", {"patient_id": "001", "findings": "Synthetic findings",
        "impression": "Synthetic impression", "technique": "Synthetic technique"})
    hashes = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
    receipt = prepare(source, output)
    assert receipt["requests"] == 1
    req = strict_loads((output / "v2/requests.jsonl").read_text())
    assert req["case_id"] == "001" and req["levels"][3]["gradings"]["disc_bulging"]["value"] is None
    assert {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()} == hashes
    with pytest.raises(ValueError): prepare(source, source / "unsafe-output")
