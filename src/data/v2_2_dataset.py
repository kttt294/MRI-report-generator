"""Doctor-authored Vietnamese report targets for the V2-2 pilot.

This dataset is not the catalog-approved Phase 1 dataset. Reports can contain
observations absent from the eight grading fields; the pilot must be evaluated
for unsupported claims before clinical use.
"""
import json
from pathlib import Path

from src.contracts.report_input import ReportRequest
from src.data.report_targets import vi_sections
from src.data.v2_adapter import DatasetAdapter
from src.data.v2_dataset import TextCollator
from src.io_utils import canonical_json, content_hash, strict_loads


INSTRUCTION = (
    "Viết báo cáo MRI cột sống thắt lưng bằng tiếng Việt từ các nhãn grading bên dưới. "
    "Trả về đúng JSON gồm hai chuỗi findings và impression. "
    "Phân biệt dữ liệu thiếu với kết quả âm tính. "
    "Không tự khẳng định bên tổn thương, rễ thần kinh hay cấu trúc ngoài các nhãn được cung cấp. "
    "Nếu đầu vào không đủ căn cứ, không được suy đoán thêm."
)


def prompt_for(request: ReportRequest) -> str:
    # prompt_facts deliberately excludes case ID, report, fold, and provenance.
    return INSTRUCTION + "\n" + canonical_json(request.prompt_facts())


def load_samples(patients_path, master_csv, fold: int, split: str):
    if fold not in range(1, 6) or split not in {"train", "val", "test"}:
        raise ValueError("Invalid fold or split")
    adapter = DatasetAdapter(master_csv)
    samples, seen = [], set()
    counts = {"patients": 0, "missing_findings": 0, "missing_impression": 0, "complete": 0}
    for raw in Path(patients_path).read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            continue
        patient = strict_loads(raw)
        case_id = str(patient["patient_id"])
        if case_id in seen:
            raise ValueError(f"Duplicate patient: {case_id}")
        seen.add(case_id)
        if set(patient["folds"]) != {f"fold{i}" for i in range(1, 6)}:
            raise ValueError(f"Missing fold assignment: {case_id}")
        assigned = patient["folds"][f"fold{fold}"]
        if assigned not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {case_id}")
        if assigned != split:
            continue
        request, _ = adapter.convert(patient)
        expected = {adapter.rows[(case_id, level.level)][f"fold{fold}_split"] for level in request.levels}
        if expected != {split}:
            raise ValueError(f"CSV/JSON fold mismatch: {case_id}")
        findings, impression = vi_sections(patient.get("reports", {}).get("vi", {}))
        counts["patients"] += 1
        counts["missing_findings"] += not bool(findings)
        counts["missing_impression"] += not bool(impression)
        if not (findings and impression):
            continue
        completion = json.dumps({"findings": findings, "impression": impression}, ensure_ascii=False)
        samples.append({"case_id": case_id, "prompt": prompt_for(request),
                        "completion": completion, "input_sha256": content_hash(request.prompt_facts()),
                        "target_sha256": content_hash({"findings": findings, "impression": impression})})
        counts["complete"] += 1
    if not samples:
        raise ValueError(f"No complete V2-2 samples for fold {fold} {split}")
    return samples, counts
