"""Strict inference payload. Dataset reports and split information are forbidden."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

LEVELS = ("L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1")
FIELDS = ("pfirrmann_grade", "modic", "disc_herniation", "disc_bulging",
          "disc_narrowing", "spondylolisthesis", "up_endplate", "low_endplate")
DOMAINS = {name: tuple(range(1, 6)) if name == "pfirrmann_grade" else
           tuple(range(4)) if name == "modic" else (0, 1) for name in FIELDS}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Observation(StrictModel):
    value: StrictInt | None
    status: Literal["observed", "missing", "not_assessed", "uncertain"]
    # Version 1 has no calibrated vision output. Never fabricate a confidence.
    uncertainty: None = None

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "observed" and self.value is None:
            raise ValueError("observed requires a value")
        if self.status != "observed" and self.value is not None:
            raise ValueError("Non-observed values must be null in contract v1")
        return self


class BinaryObservation(Observation):
    value: Annotated[StrictInt, Field(ge=0, le=1)] | None


class PfirrmannObservation(Observation):
    value: Annotated[StrictInt, Field(ge=1, le=5)] | None


class ModicObservation(Observation):
    value: Annotated[StrictInt, Field(ge=0, le=3)] | None


class Grading(StrictModel):
    pfirrmann_grade: PfirrmannObservation
    modic: ModicObservation
    disc_herniation: BinaryObservation
    disc_bulging: BinaryObservation
    disc_narrowing: BinaryObservation
    spondylolisthesis: BinaryObservation
    up_endplate: BinaryObservation
    low_endplate: BinaryObservation

    @model_validator(mode="after")
    def check_domains(self):
        for name, allowed in DOMAINS.items():
            obs = getattr(self, name)
            if obs.value is not None and obs.value not in allowed:
                raise ValueError(f"{name}: expected one of {allowed}")
        return self


class Level(StrictModel):
    level: Literal["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    gradings: Grading


class Provenance(StrictModel):
    source: Literal["dataset_grading", "vision_prediction", "synthetic_fixture"]
    version: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Quality(StrictModel):
    level_mapping: Literal["unverified", "verified"]
    ontology_review: Literal["unverified", "verified"] = "unverified"


class ReportRequest(StrictModel):
    schema_version: Literal["report-input/1.0"] = "report-input/1.0"
    ontology_version: Literal["pspines-grading/1.0"] = "pspines-grading/1.0"
    scope: Literal["lumbar_grading_8_fields"] = "lumbar_grading_8_fields"
    case_id: str = Field(min_length=1, max_length=128)
    provenance: Provenance
    quality: Quality
    levels: list[Level] = Field(min_length=5, max_length=5)

    @field_validator("levels")
    @classmethod
    def canonical_levels(cls, levels):
        if {x.level for x in levels} != set(LEVELS):
            raise ValueError("Exactly five unique lumbar levels are required")
        return sorted(levels, key=lambda x: LEVELS.index(x.level))

    def prompt_facts(self):
        """Allowlist: no case ID, provenance strings, reports or folds in prompt."""
        return {"scope": self.scope, "ontology_version": self.ontology_version,
                "levels": [x.model_dump() for x in self.levels]}
