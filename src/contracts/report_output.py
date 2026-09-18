"""Generated statements plus an application envelope with explicit failures."""
from typing import Literal
from pydantic import Field
from src.contracts.report_input import StrictModel


class Statement(StrictModel):
    text: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=40)


class ReportDraft(StrictModel):
    findings: list[Statement] = Field(min_length=1, max_length=40)
    impression: list[Statement] = Field(min_length=1, max_length=40)


class Attempt(StrictModel):
    raw: str
    errors: list[str]
    truncated: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float = 0.0


class ValidationResult(StrictModel):
    passed: bool
    checker: Literal["catalog-exact/1.0"] = "catalog-exact/1.0"
    errors: list[str]


class ReportResult(StrictModel):
    schema_version: Literal["report-output/1.0"] = "report-output/1.0"
    case_id: str
    status: Literal["ok", "needs_review", "fallback", "failed"]
    findings: str
    impression: str
    limitations: list[str]
    validation: ValidationResult
    draft: ReportDraft | None
    attempts: list[Attempt]
    generation_method: Literal["template", "llm", "template_fallback", "none"]
    input_sha256: str
    model: str
    prompt_sha256: str
    elapsed_seconds: float
