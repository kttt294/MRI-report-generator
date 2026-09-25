import json

import pytest

from scripts.evaluate_v2_2 import aligned_records, parse_report, run


def test_parse_report_requires_exact_two_nonempty_sections():
    assert parse_report('{"findings":"Mô tả", "impression":"Kết luận"}') == (
        "ok", {"findings": "Mô tả", "impression": "Kết luận"})
    assert parse_report('```json\n{"findings":"F","impression":"I"}\n```')[0] == "invalid_json"
    assert parse_report('{"findings":"F", "impression":""}')[0] == "invalid_schema"
    assert parse_report('{"findings":"F", "impression":"I", "extra":1}')[0] == "invalid_schema"


def test_alignment_detects_changed_input_and_preserves_invalid_prediction():
    samples = [{"case_id": "case-1", "input_sha256": "input", "target_sha256": "target",
                "completion": json.dumps({"findings": "F", "impression": "I"})}]
    prediction = {"case_id": "case-1", "input_sha256": "input", "target_sha256": "target",
                  "raw_output": "not JSON", "parsed": None, "status": "invalid_json",
                  "hit_token_limit": False, "seconds": 1.2}
    records = aligned_records(samples, [prediction])
    assert records[0]["status"] == "invalid_json"
    assert records[0]["prediction"] is None
    with pytest.raises(ValueError, match="different input/target"):
        aligned_records(samples, [{**prediction, "input_sha256": "wrong"}])
    with pytest.raises(ValueError, match="Duplicate or unmatched"):
        aligned_records(samples, [prediction, prediction])


def test_test_split_cannot_be_limited_or_skip_metrics(tmp_path):
    for kwargs in ({"limit": 2}, {"skip_metrics": True}):
        with pytest.raises(ValueError, match="score all cases"):
            run(tmp_path, tmp_path, tmp_path / "protocol.json", tmp_path / "new", **kwargs)
