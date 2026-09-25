import json

from src.report.v2_2_guarded import generate_guarded, inspect_generated
from src.report.v2_2_quality import bounded_grouped_prompt


def _answer(text):
    return {"raw_output": text, "tokens": 10, "seconds": 0.1}


def test_initial_prompt_has_limits_and_unambiguous_grouping_rule():
    prompt = bounded_grouped_prompt("Nguồn grading")
    assert "1200" in prompt and "650" in prompt
    assert "liệt kê rõ tên từng tầng" in prompt
    assert "Nếu khác mức độ" in prompt
    assert "hai chuỗi findings và impression" in prompt


def test_guard_accepts_first_complete_report_without_retry():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return _answer('{"findings":"L4/L5: phình đĩa đệm.","impression":"Phình L4/L5."}')

    result = generate_guarded("Nguồn grading", generate)
    assert result["status"] == "ok"
    assert len(calls) == 1


def test_guard_retries_missing_impression_once():
    calls = []
    outputs = iter(['{"findings":"F"}', '{"findings":"F","impression":"I"}'])

    def generate(prompt):
        calls.append(prompt)
        return _answer(next(outputs))

    result = generate_guarded("Nguồn grading", generate)
    assert result["status"] == "ok"
    assert len(calls) == 2
    assert "thiếu hoặc sai" in calls[1]


def test_guard_rejects_after_one_failed_retry():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return _answer('{"findings":"F"}')

    result = generate_guarded("Nguồn grading", generate)
    assert result["status"] == "rejected"
    assert result["report"] is None
    assert len(calls) == 2


def test_guard_uses_explicit_template_fallback_after_one_retry():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return _answer('{"findings":"F"}')

    result = generate_guarded(
        "Nguồn grading", generate,
        fallback=lambda: {"findings": "Pfirrmann độ 2.",
                          "impression": "Pfirrmann độ 2."})
    assert result["status"] == "fallback"
    assert result["report"]["findings"] == "Pfirrmann độ 2."
    assert len(calls) == 2


def test_guard_flags_repetition_even_in_valid_json():
    raw = json.dumps({"findings": "L4/L5 có phình đĩa đệm. " * 20,
                      "impression": "Phình L4/L5."})
    assert inspect_generated(raw)["status"] == "repetitive"


def test_guard_flags_obvious_anatomic_and_language_errors():
    def report(findings):
        return json.dumps({"findings": findings, "impression": "Phình L4/L5."})

    assert inspect_generated(report("L4/L5 có tổn thương椎体."))["status"] == "invalid_script"
    assert inspect_generated(report("S1/S2 có phình đĩa đệm."))["status"] == "invalid_level"
    assert inspect_generated(report("L4/L5 có khối u."))["status"] == "unsupported_scope"
