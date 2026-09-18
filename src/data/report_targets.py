"""Canonical report text with explicit missingness and alias conflicts."""
import math


def clean_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, list):
        return "\n".join(x for v in value if (x := clean_text(v)))
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "<na>"} else text


def canonical_text(record, canonical, alias):
    current, legacy = clean_text(record.get(canonical)), clean_text(record.get(alias))
    if current and legacy and current != legacy:
        raise ValueError(f"Conflicting report aliases: {canonical} / {alias}")
    return current or legacy


def vi_sections(record, csv=False):
    if csv:
        return (canonical_text(record, "report_vi_findings", "report_vi_mota"),
                canonical_text(record, "report_vi_impression", "report_vi_ketluan"))
    return (canonical_text(record, "findings", "mo_ta"),
            canonical_text(record, "impression", "ket_luan"))


def format_target(findings, impression):
    return f"[MÔ TẢ]:\n{findings}\n\n[KẾT LUẬN]:\n{impression}"
