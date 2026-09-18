"""Fail-closed validation for controlled sentences; not an unrestricted NLP judge."""
from src.contracts.report_output import ReportDraft
from src.io_utils import strict_loads


def validate_draft(raw, plan, truncated=False):
    errors = ["generation_truncated"] if truncated else []
    try:
        draft = ReportDraft.model_validate(strict_loads(raw))
    except ValueError as exc:
        return None, errors + [f"invalid_output_schema: {exc}"]
    for section in ("findings", "impression"):
        expected = {x["evidence_ids"][0]: x["text"] for x in plan[section]}
        seen = set()
        for statement in getattr(draft, section):
            # Exact text validation is intentionally stricter than evidence IDs.
            # Free paraphrases are rejected rather than claimed to be safe.
            if len(statement.evidence_ids) != 1:
                errors.append(f"{section}: one fact per controlled sentence required")
                continue
            fid = statement.evidence_ids[0]
            if fid in seen:
                errors.append(f"{section}: duplicate_fact:{fid}")
            seen.add(fid)
            if fid not in expected:
                errors.append(f"{section}: unexpected_fact:{fid}")
            elif statement.text.strip() != expected[fid]:
                errors.append(f"{section}: unsupported_text:{fid}")
        for fid in sorted(set(expected) - seen):
            errors.append(f"{section}: omitted_fact:{fid}")
    return draft, errors


def render(draft):
    return ("\n".join(x.text for x in draft.findings),
            "\n".join(x.text for x in draft.impression))
