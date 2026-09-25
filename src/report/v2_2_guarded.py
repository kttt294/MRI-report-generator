"""Bounded V2-2 generation with one format/length retry.

This guard checks syntax, length, and obvious repetition. It cannot certify
clinical correctness or whether grouped findings remain unambiguous.
"""
import re

from src.report.v2_2_quality import (
    bounded_grouped_prompt, inspect_report, repair_prompt,
)


MAX_ATTEMPTS = 2  # Initial generation plus one retry.
MAX_REPEATED_FIVEGRAM_FRACTION = 0.85
LUMBAR_LEVELS = {"L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"}
LEVEL_PAIR = re.compile(r"(?<![A-Za-z0-9])([LS][0-9])\s*[/\-–]\s*([LS][0-9])(?![A-Za-z0-9])")
UNSUPPORTED_SCOPE_TERMS = (
    "hẹp ống sống", "hẹp lỗ liên hợp", "foraminal", "khớp chậu", "khối u",
    "tủy sống", "rễ thần kinh",
)


def repeated_fivegram_fraction(text):
    words = re.findall(r"\S+", text.lower())
    grams = [tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def inspect_generated(raw):
    gate = inspect_report(raw)
    if gate["status"] != "ok":
        return gate
    full_text = gate["report"]["findings"] + "\n" + gate["report"]["impression"]
    if re.search(r"[\u3400-\u9fff]", full_text):
        return {"status": "invalid_script", "report": gate["report"]}
    for match in LEVEL_PAIR.finditer(full_text):
        if f"{match[1]}/{match[2]}" not in LUMBAR_LEVELS:
            return {"status": "invalid_level", "report": gate["report"]}
    if any(term in full_text.lower() for term in UNSUPPORTED_SCOPE_TERMS):
        return {"status": "unsupported_scope", "report": gate["report"]}
    if repeated_fivegram_fraction(gate["report"]["findings"]) > MAX_REPEATED_FIVEGRAM_FRACTION:
        return {"status": "repetitive", "report": gate["report"]}
    return gate


def generate_guarded(original_prompt, generate):
    """Call generate(prompt) at most twice; return no report if both fail.

    `generate` must return a dict with `raw_output`, `tokens`, and `seconds`.
    Raw text is retained for private audit but must not be published with PHI.
    """
    attempts = []
    failure_status = None
    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        prompt = (bounded_grouped_prompt(original_prompt) if attempt_number == 1
                  else repair_prompt(original_prompt, failure_status))
        generated = generate(prompt)
        if not isinstance(generated, dict) or not all(
                key in generated for key in ("raw_output", "tokens", "seconds")):
            raise ValueError("Generator must return raw_output, tokens, and seconds")
        gate = inspect_generated(generated["raw_output"])
        attempts.append({"attempt": attempt_number, "status": gate["status"],
                         "tokens": generated["tokens"],
                         "seconds": generated["seconds"],
                         "raw_output": generated["raw_output"]})
        if gate["status"] == "ok":
            return {"status": "ok", "report": gate["report"], "attempts": attempts}
        failure_status = gate["status"]
    return {"status": "rejected", "report": None, "attempts": attempts}
