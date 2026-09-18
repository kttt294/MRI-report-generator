import csv
import pytest
from src.contracts.report_input import FIELDS, LEVELS, ReportRequest


@pytest.fixture
def request_data():
    return {"case_id": "synthetic-case", "provenance": {"source": "synthetic_fixture",
        "version": "test/1", "source_sha256": "a" * 64},
        "quality": {"level_mapping": "verified", "ontology_review": "verified"},
        "levels": [{"level": level, "gradings": {name: {"value": 1 if name == "pfirrmann_grade" else 0,
            "status": "observed"} for name in FIELDS}} for level in LEVELS]}


@pytest.fixture
def report_request(request_data):
    return ReportRequest.model_validate(request_data)


@pytest.fixture
def master_rows():
    return [{"patient_id": "synthetic-case", "level": level, "volume": "phantom.nii.gz",
        "report_vi_findings": "Mô tả mẫu giả lập.", "report_vi_impression": "Kết luận mẫu giả lập.",
        **{f"fold{i}_split": "train" for i in range(1, 6)},
        **{f: 1 if f == "pfirrmann_grade" else 0 for f in FIELDS}} for level in LEVELS]


@pytest.fixture
def write_master(tmp_path):
    def write(rows):
        path = tmp_path / "master.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
        return path
    return write
