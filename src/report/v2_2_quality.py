"""Deterministic format/length gate for a proposed V2-2 retry experiment.

This gate cannot judge whether a medical claim is supported by the grading.
Thresholds are above the maxima observed in the fold-1 train/validation
doctor reports; they are not selected on the test split.
"""
from src.io_utils import strict_loads

MAX_FINDINGS_CHARS = 1200
MAX_IMPRESSION_CHARS = 650


def bounded_grouped_prompt(original_prompt, findings_limit=MAX_FINDINGS_CHARS,
                           impression_limit=MAX_IMPRESSION_CHARS):
    """Ask for concise grouping without conflating different medical facts.

    The character limits are enforced by inspect_report, not by the LLM prompt.
    """
    return (
        original_prompt
        + "\n\nĐịnh dạng bắt buộc: đúng một JSON với hai chuỗi findings và impression; "
          "không viết văn bản ngoài JSON. "
        + f"findings không quá {findings_limit} ký tự; impression không quá "
          f"{impression_limit} ký tự. Viết ngắn và kết thúc ngay sau dấu ngoặc JSON cuối. "
          "Không lặp câu hoặc liệt kê một phát hiện nhiều lần. "
          "Trong findings, có thể gộp các tầng vào một câu CHỈ khi cùng loại tổn thương, "
          "cùng mức độ, cùng tính chắc chắn và cùng các thuộc tính liên quan; "
          "liệt kê rõ tên từng tầng được gộp ngay trong câu. "
          "Các tổn thương khác nhau ở cùng một tầng có thể viết chung một câu nhưng phải "
          "gắn rõ từng tổn thương với tầng đó. "
          "Nếu khác mức độ, khác thuộc tính hoặc không thể viết rõ quan hệ bệnh–tầng, "
          "hãy tách thành câu riêng. Không tự thêm bên tổn thương hay dữ kiện ngoài grading. "
          "Chỉ dùng các tầng L1/L2, L2/L3, L3/L4, L4/L5, L5/S1 và viết tiếng Việt "
          "(ngoại trừ tên thuật ngữ quốc tế). Không mô tả ống sống, lỗ liên hợp, "
          "rễ thần kinh, khớp chậu, khối u hoặc cấu trúc khác không có nhãn đầu vào. "
          "Impression chỉ tổng hợp các phát hiện đã được mô tả và có căn cứ trong grading."
    )


def repair_prompt(original_prompt, failure_status,
                  findings_limit=MAX_FINDINGS_CHARS,
                  impression_limit=MAX_IMPRESSION_CHARS):
    explanations = {
        "invalid_json": "JSON chưa hoàn chỉnh",
        "invalid_schema": "thiếu hoặc sai một trong hai phần findings, impression",
        "findings_too_long": "findings vượt giới hạn ký tự",
        "impression_too_long": "impression vượt giới hạn ký tự",
        "repetitive": "findings lặp lại quá nhiều cụm từ",
        "invalid_script": "văn bản lẫn ký tự ngoài tiếng Việt và thuật ngữ quốc tế",
        "invalid_level": "có tầng cột sống không thuộc năm tầng đầu vào",
        "unsupported_scope": "có cấu trúc hoặc bệnh ngoài phạm vi tám nhãn grading",
    }
    if failure_status not in explanations:
        raise ValueError("No repair prompt for this status")
    return (bounded_grouped_prompt(original_prompt, findings_limit, impression_limit)
            + "\n\nLần sinh trước bị loại vì " + explanations[failure_status]
            + ". Hãy tạo lại từ đầu, hoàn thành CẢ HAI phần trong giới hạn đã nêu. "
              "Không chép lại câu trả lời trước.")


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
