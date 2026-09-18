import pytest
from scripts.evaluate_v1 import overlap, bootstrap_mean, evaluate
from src.io_utils import write_jsonl
from src.generate_report import run_batch


def test_overlap_empty_exact_and_bootstrap():
    assert overlap("", "test")["rouge_l_f1"] == 0
    assert overlap("Mô tả mẫu.", "Mô tả mẫu.") == {"word_f1": 1., "rouge_l_f1": 1.}
    assert bootstrap_mean([1., 1.])["ci95"] == [1., 1.]
    assert bootstrap_mean([])["mean"] is None


def test_v1_partial_coverage_not_hidden(master_rows, write_master, tmp_path):
    master = write_master(master_rows)
    predictions = tmp_path / "empty.jsonl"
    write_jsonl(predictions, [])
    metrics = evaluate(predictions, master, tmp_path / "results", split="train")
    assert metrics["evaluated_patients"] == 0 and len(metrics["missing_predictions"]) == 1
    with pytest.raises(ValueError): evaluate(predictions, master, tmp_path / "test", split="test")


def test_batch_invalid_case_is_visible_and_does_not_drop_good_case(report_request, tmp_path):
    source = tmp_path / "requests.jsonl"
    source.write_text('{"bad": true}\n'+report_request.model_dump_json()+"\n", encoding="utf-8")
    output = tmp_path / "results.jsonl"
    assert run_batch(source, output, {"backend": "template"}) == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    source.write_text("", encoding="utf-8")
    assert run_batch(source, output, {"backend": "template"}) == 1
