"""Read-only migration from legacy patient records plus authoritative CSV."""
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.contracts.report_input import DOMAINS, FIELDS, LEVELS, ReportRequest
from src.io_utils import file_hash


def grading_value(raw, field):
    if raw is None or str(raw).strip() == "":
        return None
    if isinstance(raw, bool):
        raise ValueError(f"Boolean is not a grading integer: {field}")
    try:
        number = Decimal(str(raw))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid grading: {field}") from exc
    if not number.is_finite() or number != number.to_integral_value() or int(number) not in DOMAINS[field]:
        raise ValueError(f"Out-of-domain grading: {field}={raw}")
    return int(number)


class DatasetAdapter:
    def __init__(self, master_csv):
        # Legacy JSON alone has already lost missingness. Require the CSV.
        self.path = Path(master_csv)
        self.source_sha256 = file_hash(self.path)
        self.rows = {}
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["patient_id"], row["level"])
                if key in self.rows:
                    raise ValueError(f"Duplicate grading key: {key}")
                self.rows[key] = row

    def convert(self, patient):
        case_id = str(patient["patient_id"])
        levels = patient["levels"]
        if len(levels) != 5 or {x["level"] for x in levels} != set(LEVELS):
            raise ValueError("Legacy record must contain five unique levels")
        converted, corrections = [], []
        for item in levels:
            level = item["level"]
            row = self.rows[(case_id, level)]
            gradings = {}
            for name in FIELDS:
                value = grading_value(row[name], name)
                legacy = grading_value(item["gradings"].get(name), name)
                if value is None and legacy == 0:
                    corrections.append({"level": level, "field": name, "from": 0, "to": None,
                                        "reason": "missing_csv_imputed_zero_in_legacy_json"})
                elif value != legacy:
                    raise ValueError(f"Unexplained CSV/JSON grading conflict: {case_id}/{level}/{name}")
                gradings[name] = {"value": value, "status": "missing" if value is None else "observed", "uncertainty": None}
            converted.append({"level": level, "gradings": gradings})
        request = ReportRequest.model_validate({
            "case_id": case_id, "provenance": {"source": "dataset_grading", "version": "legacy-adapter/1.0",
                "source_sha256": self.source_sha256},
            "quality": {"level_mapping": "unverified", "ontology_review": "unverified"}, "levels": converted})
        return request, corrections
