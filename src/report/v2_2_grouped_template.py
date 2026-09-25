"""Deterministic, unambiguous grouping of the eight supplied grading fields.

This is a scope-limited fallback, not a full radiologist report. Each fact is
rendered once; grouped levels share the same field, observed value, and status.
"""
from collections import defaultdict

from src.contracts.report_input import FIELDS, LEVELS, Level
from src.io_utils import strict_loads
from src.report.planner import LABELS


def _clause(fact):
    label = LABELS[fact["field"]]
    status, value = fact["status"], fact["value"]
    if status != "observed":
        state = {"missing": "thiếu dữ liệu về", "not_assessed": "chưa đánh giá",
                 "uncertain": "chưa xác định chắc chắn"}[status]
        return f"{state} {label}"
    if fact["field"] == "pfirrmann_grade":
        return f"Pfirrmann độ {value}"
    if fact["field"] == "modic":
        return (f"biến đổi Modic type {['0', 'I', 'II', 'III'][value]}" if value
                else "không có biến đổi Modic")
    return f"{'có' if value else 'không có'} {label}"


def _facts(levels):
    facts = []
    for level in levels:
        for field in FIELDS:
            observation = getattr(level.gradings, field)
            facts.append({"id": f"{level.level}:{field}", "level": level.level,
                          "field": field, "status": observation.status,
                          "value": observation.value})
    return facts


def _join_levels(levels):
    return " và ".join([", ".join(levels[:-1]), levels[-1]]) if len(levels) > 1 else levels[0]


def _group(facts):
    signatures = defaultdict(list)
    for index, fact in enumerate(facts):
        signatures[(fact["field"], fact["status"], fact["value"])].append((index, fact))
    grouped_ids = set()
    statements = []
    for members in signatures.values():
        if len(members) < 2:
            continue
        grouped_ids.update(fact["id"] for _, fact in members)
        levels = [fact["level"] for _, fact in members]
        text = f"Tại các tầng {_join_levels(levels)}: {_clause(members[0][1])}."
        statements.append((members[0][0], {"text": text,
                                           "evidence_ids": [fact["id"] for _, fact in members]}))
    by_level = defaultdict(list)
    for index, fact in enumerate(facts):
        if fact["id"] not in grouped_ids:
            by_level[fact["level"]].append((index, fact))
    for level, members in by_level.items():
        clauses = [_clause(fact) for _, fact in members]
        statements.append((members[0][0], {
            "text": f"Tại tầng {level}: {'; '.join(clauses)}.",
            "evidence_ids": [fact["id"] for _, fact in members]}))
    return [statement for _, statement in sorted(statements, key=lambda x: x[0])]


def render_grouped_from_facts(prompt_facts):
    if (not isinstance(prompt_facts, dict)
            or set(prompt_facts) != {"scope", "ontology_version", "levels"}
            or prompt_facts["scope"] != "lumbar_grading_8_fields"
            or prompt_facts["ontology_version"] != "pspines-grading/1.0"):
        raise ValueError("Unexpected report input scope")
    levels = [Level.model_validate(row) for row in prompt_facts["levels"]]
    if len(levels) != len(LEVELS) or {level.level for level in levels} != set(LEVELS):
        raise ValueError("Exactly five unique lumbar levels are required")
    levels.sort(key=lambda row: LEVELS.index(row.level))
    facts = _facts(levels)
    findings = _group(facts)
    impression_facts = [fact for fact in facts if
                        fact["field"] == "pfirrmann_grade" or
                        fact["status"] != "observed" or fact["value"] > 0]
    impression = _group(impression_facts)
    return {"findings": "\n".join(item["text"] for item in findings),
            "impression": "\n".join(item["text"] for item in impression),
            "findings_statements": findings, "impression_statements": impression}


def render_grouped_from_prompt(original_prompt):
    marker = original_prompt.find("\n{")
    if marker < 0:
        raise ValueError("Original prompt lacks structured grading JSON")
    return render_grouped_from_facts(strict_loads(original_prompt[marker + 1:]))
