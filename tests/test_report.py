import copy
import pytest
from src.contracts.report_input import ReportRequest
from src.report.pipeline import generate_report, make_prompt
from src.report.planner import make_plan, template_draft
from src.report.backends.base import Completion
from src.io_utils import strict_loads


@pytest.mark.parametrize("value", [True, 2, "1", float("nan")])
def test_invalid_binary(request_data, value):
    request_data["levels"][0]["gradings"]["disc_bulging"]["value"] = value
    with pytest.raises(ValueError): ReportRequest.model_validate(request_data)


def test_uncertainty_domain(request_data):
    # Null is allowed
    request_data["levels"][0]["gradings"]["disc_bulging"]["uncertainty"] = None
    req = ReportRequest.model_validate(request_data)
    assert req.levels[0].gradings.disc_bulging.uncertainty is None
    # Valid float in [0.0, 1.0] is allowed
    request_data["levels"][0]["gradings"]["disc_bulging"]["uncertainty"] = 0.35
    req = ReportRequest.model_validate(request_data)
    assert req.levels[0].gradings.disc_bulging.uncertainty == 0.35
    # Values outside [0.0, 1.0] are rejected
    for invalid in (-0.1, 1.1, "high", float("nan")):
        request_data["levels"][0]["gradings"]["disc_bulging"]["uncertainty"] = invalid
        with pytest.raises(ValueError): ReportRequest.model_validate(request_data)


def test_status_extras_and_duplicate_levels(request_data):
    for change in (lambda x: x.update(reports={}),
                   lambda x: x["levels"].__setitem__(1, x["levels"][0]),
                   lambda x: x["levels"][0]["gradings"]["modic"].update(status="missing")):
        value = copy.deepcopy(request_data)
        change(value)
        with pytest.raises(ValueError): ReportRequest.model_validate(value)


def test_template_scope_and_counterfactual(report_request):
    first = generate_report(report_request)
    assert first.status == "ok" and first.validation.passed and not first.attempts
    changed = report_request.model_copy(deep=True)
    changed.levels[2].gradings.disc_bulging.value = 1
    second = generate_report(changed)
    assert sum(a != b for a, b in zip(first.findings.splitlines(), second.findings.splitlines())) == 1
    prompt = make_prompt(report_request, make_plan(report_request))
    assert report_request.case_id not in prompt and "source_sha256" not in prompt


def test_missing_is_not_negative(request_data):
    request_data["levels"][0]["gradings"]["disc_bulging"] = {"value": None, "status": "missing"}
    result = generate_report(ReportRequest.model_validate(request_data))
    assert result.status == "needs_review" and result.validation.passed
    assert generate_report(ReportRequest.model_validate(request_data), require_verified=True).status == "failed"


class Backend:
    model_id = "synthetic"
    def __init__(self, values): self.values, self.calls = iter(values), 0
    def generate(self, prompt):
        self.calls += 1
        value = next(self.values)
        if isinstance(value, Exception): raise value
        return value


def test_repair_bounded_and_raw_failure_preserved(report_request):
    good = template_draft(make_plan(report_request)).model_dump_json()
    backend = Backend([Completion("bad"), Completion(good)])
    result = generate_report(report_request, backend)
    assert result.generation_method == "llm" and backend.calls == 2
    assert result.attempts[0].errors and not result.attempts[1].errors
    backend = Backend([Completion(good, True), Completion("bad")])
    result = generate_report(report_request, backend)
    assert result.status == "fallback" and len(result.attempts) == 2
    backend = Backend([TimeoutError("test")])
    assert generate_report(report_request, backend).status == "fallback" and backend.calls == 1


def test_invented_text_and_missing_facts_rejected(report_request):
    draft = template_draft(make_plan(report_request))
    draft.findings[0].text += " Chèn ép rễ bên trái."
    draft.impression.pop()
    result = generate_report(report_request, Backend([Completion(draft.model_dump_json())]), max_repairs=0, fallback=False)
    assert result.status == "failed"
    assert any("unsupported_text" in e for e in result.attempts[0].errors)
    assert any("omitted_fact" in e for e in result.attempts[0].errors)


@pytest.mark.parametrize("raw", ['{"a": 1, "a": 2}', '{"a": NaN}'])
def test_strict_json(raw):
    with pytest.raises(ValueError): strict_loads(raw)
