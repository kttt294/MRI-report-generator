"""Deterministic format/length gate for a proposed V2-2 retry experiment.

This gate cannot judge whether a medical claim is supported by the grading.
Thresholds are above the maxima observed in the fold-1 train/validation
doctor reports; they are not selected on the test split.
"""
from src.io_utils import strict_loads

MAX_FINDINGS_CHARS = 1200
MAX_IMPRESSION_CHARS = 650


def inspect_report(raw, findings_limit=MAX_FINDINGS_CHARS,
                   impression_limit=MAX_IMPRESSION_CHARS):
    try:
        parsed = strict_loads(raw.strip())
    except (TypeError, ValueError):
        return {"status": "invalid_json", "report": None}
    if (not isinstance(parsed, dict) or set(parsed) != {"findings", "impression"}
            or any(not isinstance(parsed[key], str) or not parsed[key].strip()
                   for key in ("findings", "impression"))):
        return {"status": "invalid_schema", "report": None}
    report = {key: parsed[key].strip() for key in ("findings", "impression")}
    if len(report["findings"]) > findings_limit:
        return {"status": "findings_too_long", "report": report}
    if len(report["impression"]) > impression_limit:
        return {"status": "impression_too_long", "report": report}
    return {"status": "ok", "report": report}


def retry_prompt(original_prompt, failure_status,
                 findings_limit=MAX_FINDINGS_CHARS,
                 impression_limit=MAX_IMPRESSION_CHARS):
    if failure_status == "ok":
        raise ValueError("A valid report must not be regenerated")
    if failure_status not in {"invalid_json", "invalid_schema", "findings_too_long",
                              "impression_too_long"}:
        raise ValueError("Unknown quality-gate status")
    return (original_prompt + "\n\nYêu cầu xuất bản cuối: Trả lời ngắn gọn, chỉ một JSON hoàn chỉnh "
            "gồm findings và impression. findings không quá " + str(findings_limit) +
            " ký tự; impression không quá " + str(impression_limit) +
            " ký tự. Kết thúc JSON ngay sau impression; không lặp lại câu hoặc liệt kê "
            "các thông tin không có trong grading.")
