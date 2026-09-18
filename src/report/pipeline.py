"""Single-call report generation with one bounded repair and visible fallback."""
import time
from pathlib import Path

from src.contracts.report_input import ReportRequest
from src.contracts.report_output import Attempt, ReportResult, ValidationResult
from src.io_utils import canonical_json, content_hash
from src.report.planner import make_plan, template_draft, limitations, needs_review
from src.report.validate import validate_draft, render

PROMPT_PATH = Path(__file__).parent / "prompts" / "vi_v1.txt"


def make_prompt(request, plan):
    # No dataset record, original target, split, case ID or arbitrary provenance.
    return PROMPT_PATH.read_text(encoding="utf-8") + "\n" + canonical_json({
        "scope": request.scope, "policy_version": plan["policy_version"],
        "findings_catalog": plan["findings"], "impression_catalog": plan["impression"]})


def generate_report(request, backend=None, max_repairs=1, fallback=True, require_verified=False):
    if not isinstance(request, ReportRequest):
        request = ReportRequest.model_validate(request)
    if max_repairs not in (0, 1):
        raise ValueError("max_repairs must be 0 or 1")
    started = time.monotonic()
    plan = make_plan(request)
    prompt = make_prompt(request, plan)
    attempts = []
    draft = None
    method, errors = "none", []
    if require_verified and needs_review(request):
        errors = ["Input QC requires review; generation not attempted"]
    elif backend is None:
        draft, method = template_draft(plan), "template"
    else:
        current_prompt = prompt
        for attempt_number in range(max_repairs + 1):
            attempt_start = time.monotonic()
            completion = None
            try:
                completion = backend.generate(current_prompt)
                candidate, errors = validate_draft(completion.text, plan, completion.truncated)
            except Exception as exc:
                candidate, errors = None, [f"backend_error:{type(exc).__name__}:{exc}"]
            attempts.append(Attempt(raw=completion.text if completion else "", errors=errors,
                truncated=completion.truncated if completion else False,
                input_tokens=completion.input_tokens if completion else None,
                output_tokens=completion.output_tokens if completion else None,
                elapsed_seconds=time.monotonic() - attempt_start))
            if not errors:
                draft, method = candidate, "llm"
                break
            if completion is None:  # Repeating an unavailable model is not a repair.
                break
            current_prompt = prompt + "\nLần trước không đạt. Sinh lại từ dữ kiện gốc, sửa các lỗi:\n" + canonical_json(errors)
        if draft is None and fallback:
            draft, method = template_draft(plan), "template_fallback"
    if draft is not None:
        # Validate template fallback as well; raw failures remain in attempts.
        _, final_errors = validate_draft(draft.model_dump_json(), plan)
        if final_errors:
            raise RuntimeError(f"Internal renderer violation: {final_errors}")
        findings, impression = render(draft)
        status = "fallback" if method == "template_fallback" else "needs_review" if needs_review(request) else "ok"
    else:
        findings, impression, status, final_errors = "", "", "failed", errors
    return ReportResult(case_id=request.case_id, status=status, findings=findings, impression=impression,
        limitations=limitations(request), draft=draft, attempts=attempts, generation_method=method,
        validation=ValidationResult(passed=draft is not None, errors=final_errors),
        input_sha256=content_hash(request.model_dump()), model=backend.model_id if backend else "deterministic-template/1.0",
        prompt_sha256=content_hash(prompt), elapsed_seconds=time.monotonic() - started)
