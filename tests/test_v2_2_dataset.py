import csv
import json
from pathlib import Path

from src.contracts.report_input import FIELDS
from src.data.v2_2_dataset import load_samples


def test_original_report_target_is_split_bound_and_absent_from_prompt(tmp_path):
    fixture = json.loads(Path("examples/report_request.synthetic.json").read_text(encoding="utf-8"))
    patients, csv_rows = [], []
    for case_id, split, impression in (("training-only", "train", "Kết luận riêng của bác sĩ."),
                                       ("validation-only", "val", "Kết luận val."),
                                       ("incomplete", "train", "")):
        patient = {"patient_id": case_id,
                   "folds": {f"fold{i}": split for i in range(1, 6)},
                   "levels": [{"level": level["level"],
                               "gradings": {name: level["gradings"][name]["value"] for name in FIELDS}}
                              for level in fixture["levels"]],
                   "reports": {"vi": {"findings": ["Mô tả riêng của bác sĩ."],
                                      "impression": [impression] if impression else []}}}
        patients.append(patient)
        for level in fixture["levels"]:
            csv_rows.append({"patient_id": case_id, "level": level["level"],
                             "fold1_split": split,
                             **{name: level["gradings"][name]["value"] for name in FIELDS}})
    master = tmp_path / "master.csv"
    with master.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "level", "fold1_split", *FIELDS])
        writer.writeheader()
        writer.writerows(csv_rows)
    records = tmp_path / "patients.jsonl"
    records.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in patients), encoding="utf-8")

    train, counts = load_samples(records, master, 1, "train")
    val, _ = load_samples(records, master, 1, "val")
    assert counts == {"patients": 2, "missing_findings": 0, "missing_impression": 1, "complete": 1}
    assert [x["case_id"] for x in train] == ["training-only"]
    assert [x["case_id"] for x in val] == ["validation-only"]
    assert "training-only" not in train[0]["prompt"]
    assert "Mô tả riêng của bác sĩ" not in train[0]["prompt"]
    assert "findings_catalog" not in train[0]["prompt"]
    assert json.loads(train[0]["completion"]) == {
        "findings": "Mô tả riêng của bác sĩ.", "impression": "Kết luận riêng của bác sĩ."}
