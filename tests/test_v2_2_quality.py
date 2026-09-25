import json

import pytest

from scripts import v2_2_retry_validation
from src.report.v2_2_quality import inspect_report, retry_prompt


def test_quality_gate_flags_format_and_overlong_sections():
    assert inspect_report('{"findings":"F", "impression":"I"}')["status"] == "ok"
    assert inspect_report('{"findings":"F"}')["status"] == "invalid_schema"
    assert inspect_report('{"findings":')["status"] == "invalid_json"
    assert inspect_report(json.dumps({"findings": "F" * 1201, "impression": "I"}))[
        "status"] == "findings_too_long"
    assert inspect_report(json.dumps({"findings": "F", "impression": "I" * 651}))[
        "status"] == "impression_too_long"


def test_retry_changes_prompt_only_after_failure():
    prompt = "Yêu cầu gốc.\n{\"levels\": []}"
    revised = retry_prompt(prompt, "invalid_json")
    assert revised.startswith(prompt)
    assert "1200" in revised and "650" in revised
    assert "JSON hoàn chỉnh" in revised
    with pytest.raises(ValueError, match="must not"):
        retry_prompt(prompt, "ok")


def test_retry_validation_cli_maps_protocol_without_test_split(monkeypatch, capsys):
    called = {}
    monkeypatch.setattr(v2_2_retry_validation, "run",
                        lambda **kwargs: called.update(kwargs) or {"ok": True})
    v2_2_retry_validation.main(["--annotations-root", "annotations", "--adapter-root", "adapter",
                                "--protocol", "protocol.json", "--output", "out", "--limit", "2"])
    assert called["protocol_path"] == "protocol.json"
    assert called["limit"] == 2
    assert "ok" in capsys.readouterr().out
