from src.contracts.report_input import FIELDS, LEVELS, ReportRequest
from src.report.v2_2_grouped_template import render_grouped_from_facts


def test_template_groups_identical_facts_and_covers_every_field(report_request):
    result = render_grouped_from_facts(report_request.prompt_facts())
    assert "L1/L2, L2/L3, L3/L4, L4/L5 và L5/S1" in result["findings"]
    evidence = [fid for row in result["findings_statements"] for fid in row["evidence_ids"]]
    assert len(evidence) == len(LEVELS) * len(FIELDS)
    assert len(set(evidence)) == len(evidence)
    assert "thoát vị" not in result["impression"]  # All binary labels are zero.


def test_template_separates_different_findings_and_missing_values(request_data):
    request_data["levels"][0]["gradings"]["disc_bulging"]["value"] = 1
    request_data["levels"][1]["gradings"]["disc_bulging"]["value"] = 1
    request_data["levels"][1]["gradings"]["disc_herniation"]["value"] = 1
    request_data["levels"][2]["gradings"]["disc_bulging"] = {
        "value": None, "status": "missing"}
    result = render_grouped_from_facts(ReportRequest.model_validate(request_data).prompt_facts())
    assert "L1/L2 và L2/L3: có phình đĩa đệm" in result["findings"]
    assert "Tại tầng L2/L3: có thoát vị đĩa đệm" in result["findings"]
    assert "Tại tầng L3/L4: thiếu dữ liệu về phình đĩa đệm" in result["findings"]
    assert "L3/L4: không có phình" not in result["findings"]
