"""Deterministic facts and a controlled sentence catalog (not clinical adjudication)."""
from src.contracts.report_input import FIELDS
from src.contracts.report_output import ReportDraft, Statement

POLICY_VERSION = "grading-verbatim/1.0"
LABELS = {
    "pfirrmann_grade": "phân độ Pfirrmann", "modic": "biến đổi Modic",
    "disc_herniation": "thoát vị đĩa đệm", "disc_bulging": "phình đĩa đệm",
    "disc_narrowing": "giảm chiều cao khe đĩa đệm", "spondylolisthesis": "trượt đốt sống",
    "up_endplate": "bất thường bản đệm trên", "low_endplate": "bất thường bản đệm dưới",
}


def fact_text(level, field, observation):
    label = LABELS[field]
    prefix = f"Tại tầng {level}: "
    if observation.status != "observed":
        state = {"missing": "thiếu dữ liệu về", "not_assessed": "chưa đánh giá",
                 "uncertain": "chưa xác định chắc chắn"}[observation.status]
        return f"{prefix}{state} {label}."
    value = observation.value
    if field == "pfirrmann_grade":
        return f"{prefix}Pfirrmann độ {value}."
    if field == "modic":
        return f"{prefix}biến đổi Modic type {['0', 'I', 'II', 'III'][value]}." if value else f"{prefix}không có biến đổi Modic."
    # Do not infer laterality, morphology, cause, severity or nerve effects.
    return f"{prefix}{'có' if value else 'không có'} {label}."


def make_plan(request):
    facts, findings, impression = [], [], []
    for level in request.levels:
        for field in FIELDS:
            obs = getattr(level.gradings, field)
            fid = f"{level.level}:{field}"
            text = fact_text(level.level, field, obs)
            facts.append({"id": fid, "level": level.level, "field": field, **obs.model_dump(), "text": text})
            statement = {"text": text, "evidence_ids": [fid]}
            findings.append(statement)
            # Conservative summary: retain raw Pfirrmann, positives and unresolved
            # observations. No severity ranking or inference that grade 2 is disease.
            if field == "pfirrmann_grade" or obs.status != "observed" or obs.value > 0:
                impression.append(statement)
    return {"policy_version": POLICY_VERSION, "facts": facts,
            "findings": findings, "impression": impression}


def template_draft(plan):
    return ReportDraft(findings=[Statement(**x) for x in plan["findings"]],
                       impression=[Statement(**x) for x in plan["impression"]])


def limitations(request):
    result = ["Chỉ mô tả tám nhãn grading được cung cấp; không phải báo cáo MRI toàn diện.",
              "Độ trung thành với JSON không xác nhận nhãn đúng trên MRI."]
    if request.quality.level_mapping != "verified":
        result.append("Ánh xạ nhãn grading sang tầng chưa được xác minh.")
    if request.quality.ontology_review != "verified":
        result.append("Quy ước thuật ngữ và bản đệm/trượt đốt sống còn cần người gán nhãn xác nhận.")
    if any(getattr(level.gradings, field).status != "observed" for level in request.levels for field in FIELDS):
        result.append("Có nhãn thiếu, chưa đánh giá hoặc chưa chắc chắn; không được coi là âm tính.")
    return result


def needs_review(request):
    return (request.quality.level_mapping != "verified" or request.quality.ontology_review != "verified" or
            any(getattr(x.gradings, f).status != "observed" for x in request.levels for f in FIELDS))
