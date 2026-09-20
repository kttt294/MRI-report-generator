"""
Script: consolidate_dataset.py
Mục đích: Tự động tổng hợp và làm sạch toàn bộ dữ liệu phân mảnh từ thư mục `dataset_local/`
vào thư mục chuẩn `dataset/`, tạo ra các tệp dữ liệu tinh gọn, sẵn sàng cho huấn luyện mô hình (ML/LLM-ready).

Các tệp được tạo ra trong `dataset/`:
1. dataset_master.csv: Bảng tổng hợp chi tiết theo từng tầng đĩa đệm (1,235 dòng: 247 ca x 5 tầng).
2. dataset_patients.jsonl: Dữ liệu cấu trúc cấp bệnh nhân (247 dòng), chứa đầy đủ metadata, 5 tầng grading, toạ độ, fold split và báo cáo Việt/Anh.
Không sinh SFT legacy hoặc bản sao một bệnh nhân. Target train V2 được chuẩn bị
riêng bằng build_v2_targets.py và cần người chuyên môn duyệt.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contracts.report_input import DOMAINS, LEVELS
from src.data.report_targets import canonical_text, vi_sections, clean_text


def consolidate(source="dataset_local", output="dataset"):
    # Fix Windows console encoding for Vietnamese characters
    if sys.stdout.encoding != 'utf-8' and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')

    BASE_DIR = Path(source).resolve()
    OUT_DIR = Path(output).resolve()
    if OUT_DIR == BASE_DIR or OUT_DIR.is_relative_to(BASE_DIR):
        raise ValueError("ETL output must be outside the read-only source directory")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BẮT ĐẦU QUY TRÌNH HỢP NHẤT DỮ LIỆU PSPINES...")
    print("=" * 60)

    # 1. NẠP DỮ LIỆU GRADING
    print("\n[1/6] Nạp dữ liệu nhãn bệnh lý (grading_all.csv)...")
    df_grading = pd.read_csv(BASE_DIR / "grading" / "grading_all.csv", dtype={"Patient ID": str})
    df_grading["patient_id"] = df_grading["Patient ID"].astype(str)
    df_grading = df_grading.rename(columns={
        "IVD label": "ivd_label",
        "Modic": "modic",
        "UP endplate": "up_endplate",
        "LOW endplate": "low_endplate",
        "Spondylolisthesis": "spondylolisthesis",
        "Disc herniation": "disc_herniation",
        "Disc narrowing": "disc_narrowing",
        "Disc bulging": "disc_bulging",
        "Pfirrman grade": "pfirrmann_grade"
    })
    grading_cols = ["patient_id", "level", "ivd_label", "modic", "up_endplate", "low_endplate",
                    "spondylolisthesis", "disc_herniation", "disc_narrowing", "disc_bulging",
                    "pfirrmann_grade", "month"]
    df_grading = df_grading[grading_cols]
    print(f" -> Đã nạp {len(df_grading)} dòng nhãn ({df_grading['patient_id'].nunique()} bệnh nhân).")

    # 2. NẠP DỮ LIỆU ĐỊNH VỊ TOẠ ĐỘ (disc_localization.csv)
    print("\n[2/6] Nạp dữ liệu toạ độ tâm đĩa đệm (disc_localization.csv)...")
    df_loc = pd.read_csv(BASE_DIR / "localize" / "disc_localization.csv", dtype={"patient_id": str})
    df_loc["patient_id"] = df_loc["patient_id"].astype(str)
    # Chỉ lấy các ca thuộc cohort phân tích
    df_loc = df_loc[df_loc["in_cohort"] == 1].copy()
    loc_cols = ["patient_id", "level", "i", "j", "k", "x_lps", "y_lps", "z_lps",
                "volume", "spacing_i", "spacing_j", "spacing_k", "source", "qc_status"]
    df_loc = df_loc[loc_cols].rename(columns={
        "i": "voxel_i", "j": "voxel_j", "k": "voxel_k",
        "source": "loc_source", "qc_status": "loc_qc_status"
    })
    print(f" -> Đã nạp toạ độ {len(df_loc)} đĩa đệm ({df_loc['patient_id'].nunique()} bệnh nhân).")

    # 3. NẠP THÔNG TIN 5-FOLD CROSS VALIDATION
    print("\n[3/6] Nạp thông tin phân chia 5-fold cross-validation...")
    fold_splits = {f"fold{i}_split": {} for i in range(1, 6)}

    for fold_idx in range(1, 6):
        fold_dir = BASE_DIR / "folds" / f"fold{fold_idx}"
        for split in ["train", "val", "test"]:
            split_file = fold_dir / f"{split}.csv"
            if split_file.exists():
                df_split = pd.read_csv(split_file, dtype={"Patient ID": str})
                pts = df_split["Patient ID"].astype(str).unique()
                for p in pts:
                    if p in fold_splits[f"fold{fold_idx}_split"]:
                        raise ValueError(f"Duplicate patient across fold {fold_idx} split files")
                    fold_splits[f"fold{fold_idx}_split"][p] = split

    fold_df = pd.DataFrame(fold_splits)
    fold_df.index.name = "patient_id"
    fold_df = fold_df.reset_index()
    print(f" -> Đã liên kết 5 fold cho {len(fold_df)} bệnh nhân.")

    # 4. NẠP METADATA VÀ BÁO CÁO GỐC TIẾNG VIỆT (reports_json)
    print("\n[4/6] Nạp metadata nhân khẩu học và báo cáo tiếng Việt...")
    meta_dir = BASE_DIR / "reports_json" / "metadata"
    report_dir = BASE_DIR / "reports_json" / "report"

    meta_records = {}
    if meta_dir.exists():
        for f in os.listdir(meta_dir):
            if f.endswith(".json"):
                with open(meta_dir / f, encoding="utf-8") as jf:
                    data = json.load(jf)
                    pid = str(data["patient_id"])
                    meta_records[pid] = {
                        "sub_id": data.get("sub_id", ""),
                        "sex": data.get("sex", ""),
                        "birth_year": data.get("birth_year", None),
                        "age_at_scan": data.get("age_at_scan", None),
                        "study_date": str(data.get("study_date", ""))
                    }

    vi_reports = {}
    if report_dir.exists():
        for f in os.listdir(report_dir):
            if f.endswith(".json"):
                with open(report_dir / f, encoding="utf-8") as jf:
                    data = json.load(jf)
                    pid = str(data["patient_id"])
                    if pid in vi_reports:
                        raise ValueError("Duplicate source report patient")
                    findings, impression = vi_sections(data)
                    mota_list = findings.splitlines() if findings else []
                    ketluan_list = impression.splitlines() if impression else []
                    vi_reports[pid] = {
                        "ky_thuat": canonical_text(data, "technique", "ky_thuat"),
                        "mo_ta_list": mota_list,
                        "ket_luan_list": ketluan_list,
                        "report_vi_mota": "\n".join(mota_list),
                        "report_vi_ketluan": "\n".join(ketluan_list)
                    }

    print(f" -> Đã nạp metadata cho {len(meta_records)} ca, báo cáo tiếng Việt cho {len(vi_reports)} ca.")

    # 5. NẠP BÁO CÁO TIẾNG ANH (reports_text)
    print("\n[5/6] Nạp báo cáo tiếng Anh (reports_text)...")
    sub_to_pid = {v["sub_id"]: k for k, v in meta_records.items() if v.get("sub_id")}

    en_reports = {}
    text_dir = BASE_DIR / "reports_text"
    if text_dir.exists():
        for fname in ["train.csv", "val.csv", "test.csv", "missax2.csv"]:
            fpath = text_dir / fname
            if fpath.exists():
                df_txt = pd.read_csv(fpath)
                for _, row in df_txt.iterrows():
                    sub = str(row["sub_id"])
                    pid = sub_to_pid.get(sub)
                    if pid:
                        en_reports[pid] = {
                            "report_en": clean_text(row.get("Clinician's Notes", "")),
                            "reports_text_split": clean_text(row.get("split", "")),
                            "case_id": clean_text(row.get("case_id", ""))
                        }

    print(f" -> Đã nạp báo cáo tiếng Anh cho {len(en_reports)} ca.")

    # 6. GỘP TOÀN BỘ VÀO DATASET_MASTER.CSV
    print("\n[6/6] Đang tiến hành kết hợp dữ liệu vào dataset_master.csv và dataset_patients.jsonl...")

    # Gộp Grading + Localization
    df_master = pd.merge(df_grading, df_loc, on=["patient_id", "level"], how="left", validate="one_to_one")

    # Gộp Folds
    df_master = pd.merge(df_master, fold_df, on="patient_id", how="left", validate="many_to_one")

    # Thêm metadata nhân khẩu
    df_master["sub_id"] = df_master["patient_id"].map(lambda p: meta_records.get(p, {}).get("sub_id", ""))
    df_master["sex"] = df_master["patient_id"].map(lambda p: meta_records.get(p, {}).get("sex", ""))
    df_master["age_at_scan"] = df_master["patient_id"].map(lambda p: meta_records.get(p, {}).get("age_at_scan", None))
    df_master["birth_year"] = df_master["patient_id"].map(lambda p: meta_records.get(p, {}).get("birth_year", None))
    df_master["study_date"] = df_master["patient_id"].map(lambda p: meta_records.get(p, {}).get("study_date", ""))

    # Thêm báo cáo tiếng Việt (Chuẩn hóa tiếng Anh + Giữ alias tương thích ngược)
    df_master["report_vi_technique"] = df_master["patient_id"].map(lambda p: vi_reports.get(p, {}).get("ky_thuat", ""))
    df_master["report_vi_findings"] = df_master["patient_id"].map(lambda p: vi_reports.get(p, {}).get("report_vi_mota", ""))
    df_master["report_vi_impression"] = df_master["patient_id"].map(lambda p: vi_reports.get(p, {}).get("report_vi_ketluan", ""))

    # Alias tương thích ngược cho các script cũ
    df_master["report_vi_kythuat"] = df_master["report_vi_technique"]
    df_master["report_vi_mota"] = df_master["report_vi_findings"]
    df_master["report_vi_ketluan"] = df_master["report_vi_impression"]

    # Thêm báo cáo tiếng Anh
    df_master["report_en"] = df_master["patient_id"].map(lambda p: en_reports.get(p, {}).get("report_en", ""))
    df_master["reports_text_split"] = df_master["patient_id"].map(lambda p: en_reports.get(p, {}).get("reports_text_split", ""))

    # Verify domains and patient-level folds before writing anything.
    for field, allowed in DOMAINS.items():
        if not df_master[field].dropna().isin(allowed).all():
            raise ValueError(f"Invalid grading domain: {field}")
    for pid, group in df_master.groupby("patient_id"):
        if len(group) != 5 or set(group["level"]) != set(LEVELS):
            raise ValueError("Expected five unique lumbar levels per patient")
        for fold in range(1, 6):
            values = set(group[f"fold{fold}_split"])
            if len(values) != 1 or not values <= {"train", "val", "test"}:
                raise ValueError(f"Invalid fold {fold}")
    # Xuất dataset_master.csv
    master_csv_path = OUT_DIR / "dataset_master.csv"
    df_master.to_csv(master_csv_path, index=False, encoding="utf-8")
    print(f" -> Đã lưu: {master_csv_path} ({len(df_master)} dòng, {len(df_master.columns)} cột).")

    # 7. XUẤT DATASET_PATIENTS.JSONL (Dạng cấu trúc cấp bệnh nhân)
    patient_level_data = []
    all_patient_ids = df_master["patient_id"].unique()

    for pid in all_patient_ids:
        p_df = df_master[df_master["patient_id"] == pid]
        first_row = p_df.iloc[0]

        levels_info = []
        for _, row in p_df.iterrows():
            levels_info.append({
                "level": row["level"],
                "ivd_label": int(row["ivd_label"]) if pd.notna(row["ivd_label"]) else None,
                "gradings": {
                    "pfirrmann_grade": int(row["pfirrmann_grade"]) if pd.notna(row["pfirrmann_grade"]) else None,
                    "modic": int(row["modic"]) if pd.notna(row["modic"]) else None,
                    "disc_herniation": int(row["disc_herniation"]) if pd.notna(row["disc_herniation"]) else None,
                    "disc_bulging": int(row["disc_bulging"]) if pd.notna(row["disc_bulging"]) else None,
                    "disc_narrowing": int(row["disc_narrowing"]) if pd.notna(row["disc_narrowing"]) else None,
                    "spondylolisthesis": int(row["spondylolisthesis"]) if pd.notna(row["spondylolisthesis"]) else None,
                    "up_endplate": int(row["up_endplate"]) if pd.notna(row["up_endplate"]) else None,
                    "low_endplate": int(row["low_endplate"]) if pd.notna(row["low_endplate"]) else None,
                },
                "coordinates": {
                    "voxel_i": float(row["voxel_i"]) if pd.notna(row["voxel_i"]) else None,
                    "voxel_j": float(row["voxel_j"]) if pd.notna(row["voxel_j"]) else None,
                    "voxel_k": float(row["voxel_k"]) if pd.notna(row["voxel_k"]) else None,
                    "x_lps": float(row["x_lps"]) if pd.notna(row["x_lps"]) else None,
                    "y_lps": float(row["y_lps"]) if pd.notna(row["y_lps"]) else None,
                    "z_lps": float(row["z_lps"]) if pd.notna(row["z_lps"]) else None
                }
            })

        patient_entry = {
            "patient_id": pid,
            "sub_id": first_row["sub_id"],
            "demographics": {
                "sex": first_row["sex"],
                "age_at_scan": int(first_row["age_at_scan"]) if pd.notna(first_row["age_at_scan"]) else None,
                "birth_year": int(first_row["birth_year"]) if pd.notna(first_row["birth_year"]) else None,
                "study_date": first_row["study_date"]
            },
            "folds": {
                f"fold{i}": first_row.get(f"fold{i}_split", "") for i in range(1, 6)
            },
            "levels": levels_info,
            "reports": {
                "vi": {
                    "technique": vi_reports.get(pid, {}).get("ky_thuat", ""),
                    "findings": vi_reports.get(pid, {}).get("mo_ta_list", []),
                    "impression": vi_reports.get(pid, {}).get("ket_luan_list", [])
                },
                "en": {
                    "clinicians_notes": en_reports.get(pid, {}).get("report_en", ""),
                    "split": en_reports.get(pid, {}).get("reports_text_split", "")
                }
            }
        }
        patient_level_data.append(patient_entry)

    patients_jsonl_path = OUT_DIR / "dataset_patients.jsonl"
    with open(patients_jsonl_path, "w", encoding="utf-8") as f:
        for item in patient_level_data:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    print(f" -> Đã lưu: {patients_jsonl_path} ({len(patient_level_data)} ca bệnh nhân).")

    # Only the two shared dataset files are exported. V2 training targets are
    # prepared separately with build_v2_targets.py after human review.

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT HỢP NHẤT DỮ LIỆU! Toàn bộ file đã được tạo tại: {OUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="dataset_local")
    parser.add_argument("--output", default="dataset")
    args = parser.parse_args()
    consolidate(args.source, args.output)
