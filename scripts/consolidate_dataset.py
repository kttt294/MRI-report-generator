"""
Script: consolidate_dataset.py
Mục đích: Tự động tổng hợp và làm sạch toàn bộ dữ liệu phân mảnh từ thư mục `dataset_local/`
vào thư mục chuẩn `dataset/`, tạo ra các tệp dữ liệu tinh gọn, sẵn sàng cho huấn luyện mô hình (ML/LLM-ready).

Các tệp được tạo ra trong `dataset/`:
1. dataset_master.csv: Bảng tổng hợp chi tiết theo từng tầng đĩa đệm (1,235 dòng: 247 ca x 5 tầng).
2. dataset_patients.jsonl: Dữ liệu cấu trúc cấp bệnh nhân (247 dòng), chứa đầy đủ metadata, 5 tầng grading, toạ độ, fold split và báo cáo Việt/Anh.
3. sft_fold1_train_vi.jsonl, sft_fold1_val_vi.jsonl, sft_fold1_test_vi.jsonl: Bộ dữ liệu Prompt-Response SFT tiếng Việt (Fold 1).
4. sft_fold1_train_en.jsonl, sft_fold1_val_en.jsonl, sft_fold1_test_en.jsonl: Bộ dữ liệu Prompt-Response SFT tiếng Anh (Fold 1).
5. README.md: Tài liệu hướng dẫn sử dụng và lược đồ dữ liệu (schema).
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Fix Windows console encoding for Vietnamese characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path("dataset_local")
OUT_DIR = Path("dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("BẮT ĐẦU QUY TRÌNH HỢP NHẤT DỮ LIỆU PSPINES...")
print("=" * 60)

# 1. NẠP DỮ LIỆU GRADING
print("\n[1/6] Nạp dữ liệu nhãn bệnh lý (grading_all.csv)...")
df_grading = pd.read_csv(BASE_DIR / "grading" / "grading_all.csv")
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
df_loc = pd.read_csv(BASE_DIR / "localize" / "disc_localization.csv")
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
            df_split = pd.read_csv(split_file)
            pts = df_split["Patient ID"].astype(str).unique()
            for p in pts:
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
                mota_list = data.get("mo_ta", [])
                ketluan_list = data.get("ket_luan", [])
                vi_reports[pid] = {
                    "ky_thuat": data.get("ky_thuat", ""),
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
                        "report_en": row.get("Clinician's Notes", ""),
                        "reports_text_split": row.get("split", ""),
                        "case_id": row.get("case_id", "")
                    }

print(f" -> Đã nạp báo cáo tiếng Anh cho {len(en_reports)} ca.")

# 6. GỘP TOÀN BỘ VÀO DATASET_MASTER.CSV
print("\n[6/6] Đang tiến hành kết hợp dữ liệu vào dataset_master.csv và dataset_patients.jsonl...")

# Gộp Grading + Localization
df_master = pd.merge(df_grading, df_loc, on=["patient_id", "level"], how="left")

# Gộp Folds
df_master = pd.merge(df_master, fold_df, on="patient_id", how="left")

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
                "modic": int(row["modic"]) if pd.notna(row["modic"]) else 0,
                "disc_herniation": int(row["disc_herniation"]) if pd.notna(row["disc_herniation"]) else 0,
                "disc_bulging": int(row["disc_bulging"]) if pd.notna(row["disc_bulging"]) else 0,
                "disc_narrowing": int(row["disc_narrowing"]) if pd.notna(row["disc_narrowing"]) else 0,
                "spondylolisthesis": int(row["spondylolisthesis"]) if pd.notna(row["spondylolisthesis"]) else 0,
                "up_endplate": int(row["up_endplate"]) if pd.notna(row["up_endplate"]) else 0,
                "low_endplate": int(row["low_endplate"]) if pd.notna(row["low_endplate"]) else 0,
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
                "impression": vi_reports.get(pid, {}).get("ket_luan_list", []),
                # Giữ alias tương thích ngược
                "ky_thuat": vi_reports.get(pid, {}).get("ky_thuat", ""),
                "mo_ta": vi_reports.get(pid, {}).get("mo_ta_list", []),
                "ket_luan": vi_reports.get(pid, {}).get("ket_luan_list", [])
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
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f" -> Đã lưu: {patients_jsonl_path} ({len(patient_level_data)} ca bệnh nhân).")

# Lưu tệp mẫu 1 bệnh nhân với cấu trúc đẹp mắt
sample_patient_path = OUT_DIR / "data_of_1patient.json"
with open(sample_patient_path, "w", encoding="utf-8") as f:
    json.dump(patient_level_data[0], f, ensure_ascii=False, indent=2)
print(f" -> Đã lưu mẫu chuẩn: {sample_patient_path}")

# 8. TẠO DATASET HUẤN LUYỆN SFT CHO FOLD 1 (Prompt -> Response)
def format_vi_prompt(levels_info, demo):
    age_str = f"Tuổi: {demo['age_at_scan']}, " if demo.get("age_at_scan") else ""
    sex_str = f"Giới tính: {demo['sex']}\n" if demo.get("sex") else "\n"
    prompt = f"[THÔNG TIN BỆNH NHÂN]: {age_str}{sex_str}"
    prompt += "[KẾT QUẢ KHẢO SÁT 5 TẦNG ĐĨA ĐỆM CỘT SỐNG THẮT LƯNG]:\n"
    for lvl in levels_info:
        g = lvl["gradings"]
        name = lvl["level"]
        details = []
        if g.get("pfirrmann_grade"): details.append(f"Thoái hóa Pfirrmann độ {g['pfirrmann_grade']}")
        if g.get("disc_herniation"): details.append("Có thoát vị đĩa đệm")
        if g.get("disc_bulging"): details.append("Có phình đĩa đệm")
        if g.get("disc_narrowing"): details.append("Có hẹp khe đĩa đệm")
        if g.get("spondylolisthesis"): details.append("Có trượt đốt sống")
        if g.get("modic"): details.append("Có thoái hóa Modic")
        if not details: details.append("Bình thường, không tổn thương rõ")
        prompt += f"- Tầng {name}: {', '.join(details)}.\n"
    prompt += "\n[YÊU CẦU]: Dựa trên các phát hiện bệnh lý trên, hãy viết phần MÔ TẢ và KẾT LUẬN báo cáo cộng hưởng từ cột sống thắt lưng."
    return prompt

def format_vi_response(rep_vi):
    findings_list = rep_vi.get("findings", rep_vi.get("mo_ta", []))
    impression_list = rep_vi.get("impression", rep_vi.get("ket_luan", []))
    mota = "\n".join(f"- {c}" for c in findings_list)
    ketluan = "\n".join(f"- {c}" for c in impression_list)
    return f"[MÔ TẢ]:\n{mota}\n\n[KẾT LUẬN]:\n{ketluan}"

def format_en_prompt(levels_info, demo):
    age_str = f"Age: {demo['age_at_scan']}, " if demo.get("age_at_scan") else ""
    sex_str = f"Sex: {demo['sex']}\n" if demo.get("sex") else "\n"
    prompt = f"[PATIENT INFORMATION]: {age_str}{sex_str}"
    prompt += "[LUMBAR SPINE FINDINGS (L1-S1)]:\n"
    for lvl in levels_info:
        g = lvl["gradings"]
        name = lvl["level"]
        details = []
        if g.get("pfirrmann_grade"): details.append(f"Pfirrmann Grade {g['pfirrmann_grade']}")
        if g.get("disc_herniation"): details.append("Disc Herniation")
        if g.get("disc_bulging"): details.append("Disc Bulging")
        if g.get("disc_narrowing"): details.append("Disc Narrowing")
        if g.get("spondylolisthesis"): details.append("Spondylolisthesis")
        if g.get("modic"): details.append("Modic change")
        if not details: details.append("Normal limits")
        prompt += f"- Level {name}: {', '.join(details)}.\n"
    prompt += "\n[INSTRUCTION]: Generate the complete Clinician's Notes based on these findings."
    return prompt

# Tạo thư mục con sft_data
sft_dir = OUT_DIR / "sft_data"
sft_dir.mkdir(exist_ok=True)

for split in ["train", "val", "test"]:
    sft_vi_items = []
    sft_en_items = []
    
    for p in patient_level_data:
        if p["folds"].get("fold1") == split:
            # Tiếng Việt
            rep_vi = p["reports"]["vi"]
            if rep_vi.get("mo_ta") or rep_vi.get("ket_luan"):
                prompt_vi = format_vi_prompt(p["levels"], p["demographics"])
                resp_vi = format_vi_response(rep_vi)
                sft_vi_items.append({"patient_id": p["patient_id"], "prompt": prompt_vi, "response": resp_vi})
            
            # Tiếng Anh
            rep_en = p["reports"]["en"]
            if rep_en.get("clinicians_notes"):
                prompt_en = format_en_prompt(p["levels"], p["demographics"])
                resp_en = rep_en["clinicians_notes"]
                sft_en_items.append({"patient_id": p["patient_id"], "prompt": prompt_en, "response": resp_en})
    
    # Ghi file
    with open(sft_dir / f"sft_fold1_{split}_vi.jsonl", "w", encoding="utf-8") as f:
        for it in sft_vi_items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(sft_dir / f"sft_fold1_{split}_en.jsonl", "w", encoding="utf-8") as f:
        for it in sft_en_items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    
    print(f" -> Fold 1 [{split}]: Tạo xong {len(sft_vi_items)} mẫu SFT tiếng Việt, {len(sft_en_items)} mẫu tiếng Anh.")

print("\n" + "=" * 60)
print(f"HOÀN TẤT HỢP NHẤT DỮ LIỆU! Toàn bộ file đã được tạo tại: {OUT_DIR.resolve()}")
print("=" * 60)
