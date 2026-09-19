"""Reproducible, aggregate-only EDA. CPU only; source files are never modified.

Run: python scripts/build_complete_eda.py --output outputs/eda_2026-09-18
Requires pandas. Annotation flags reuse the project's heuristic audit, not a
clinical adjudication. Private audit rows remain in the ignored output folder.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from audit_annotation_consistency import FIELDS, extract, state as report_state, run as annotation_audit

DOMAINS = {"pfirrmann_grade": list(range(1, 6)), "modic": list(range(4)),
           **{k: [0, 1] for k in ("disc_herniation", "disc_bulging", "disc_narrowing",
                                  "spondylolisthesis", "up_endplate", "low_endplate")}}
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def build(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    source = root / "dataset/dataset_master.csv"
    # Keep strings to avoid interpreting identifiers and dates as measurements.
    df = pd.read_csv(source, dtype=str, keep_default_na=False).apply(lambda s: s.str.strip())
    p = df.drop_duplicates("patient_id").copy()
    n, nr = len(p), len(df)
    tables = []

    def table(sheet, title, headers, rows, note="", ratio=None):
        cleaned = [[x.item() if hasattr(x, "item") else x for x in row] for row in rows]
        tables.append(dict(sheet=sheet, title=title, headers=headers, rows=cleaned,
                           note=note, ratio=ratio))

    def numeric(series):
        return pd.to_numeric(series.replace("", None), errors="coerce")

    overview = [["Bệnh nhân / ca trong master", n, "bệnh nhân"],
                ["Bản ghi patient–level", nr, "tầng đĩa đệm"],
                ["Trường dữ liệu", len(df.columns), "cột"],
                ["Tầng chuẩn mỗi bệnh nhân", 5, "tầng"],
                ["Bệnh nhân có findings và impression VI", int(((p.report_vi_findings != "") & (p.report_vi_impression != "")).sum()), "bệnh nhân"]]
    table("Tong quan", "Quy mô dữ liệu đang có", ["Chỉ tiêu", "Số lượng", "Đơn vị"], overview,
          "Snapshot bảng local; không suy ra số ảnh MRI thực có trên cloud.")
    rows = []
    for col in df:
        miss = df[col].eq("")
        rows.append([col, int(miss.sum()), nr, None,
                     df.loc[miss, "patient_id"].nunique(), n,
                     df.loc[~miss, col].nunique()])
    table("Thieu du lieu", "Thiếu dữ liệu: toàn bộ các cột", ["Trường", "Dòng thiếu", "Tổng dòng", "Tỷ lệ dòng thiếu", "BN có ít nhất 1 dòng thiếu", "Tổng BN", "Giá trị khác nhau (có dữ liệu)"], rows,
          "Dòng = một tầng đĩa đệm; thông tin bệnh nhân/báo cáo lặp trên 5 dòng. Ô trống là thiếu; số 0 là giá trị.", [3, 1, 2])

    rows = []
    for col in ["sex", "month"]:
        for value, count in p[col].replace("", "Thiếu").value_counts().sort_index().items():
            rows.append([col, value, int(count), n, None])
    age = numeric(p.age_at_scan)
    bins = pd.cut(age, [-float("inf"), 20, 40, 60, 80, float("inf")], right=False,
                  labels=["<20", "20–39", "40–59", "60–79", "80+"])
    for value, count in bins.value_counts(sort=False).items():
        rows.append(["Nhóm tuổi mô tả", str(value), int(count), n, None])
    rows.append(["Nhóm tuổi mô tả", "Thiếu", int(age.isna().sum()), n, None])
    table("Nhan khau", "Nhân khẩu học và phân vùng nguồn", ["Biến", "Nhóm", "Số BN", "Tổng BN", "Tỷ lệ"], rows,
          "Đếm mỗi bệnh nhân một lần. Nhóm tuổi chỉ để mô tả; month là phân vùng nguồn, không dùng suy luận xu hướng thời gian.", [4, 2, 3])

    stats = []
    for col, series, unit in [("age_at_scan", age, "tuổi ghi trong nguồn")] + [(c, numeric(df[c]), "voxel" if c.startswith("voxel") else "mm") for c in ["voxel_i", "voxel_j", "voxel_k", "x_lps", "y_lps", "z_lps", "spacing_i", "spacing_j", "spacing_k"]]:
        valid = series.dropna()
        stats.append([col, unit, len(valid), int(series.isna().sum())] +
                     [round(float(v), 3) if pd.notna(v) else None for v in [valid.min(), valid.quantile(.25), valid.median(), valid.mean(), valid.quantile(.75), valid.max(), valid.std()]])
    table("So do", "Thống kê biến số", ["Biến", "Đơn vị", "N hợp lệ", "Thiếu / không số", "Min", "Q1", "Median", "Mean", "Q3", "Max", "SD mẫu"], stats,
          "Tuổi: cấp bệnh nhân. Tọa độ/spacing: cấp tầng, có lặp spacing theo ảnh; chưa kiểm tra bounds hay orientation trên NIfTI. Quantile nội suy tuyến tính; SD ddof=1.")

    grades = df[list(DOMAINS)].apply(numeric)
    rows = []
    for scope in ["Tất cả"] + LEVELS:
        idx = df.index if scope == "Tất cả" else df.index[df.level == scope]
        for col, domain in DOMAINS.items():
            s = grades.loc[idx, col]
            for value in domain:
                rows.append([scope, col, value, int(s.eq(value).sum()), len(s), None])
            rows.append([scope, col, "Thiếu", int(s.isna().sum()), len(s), None])
            rows.append([scope, col, "Ngoài miền", int((s.notna() & ~s.isin(domain)).sum()), len(s), None])
    table("Grading", "Phân bố từng nhãn, toàn bộ và theo tầng", ["Phạm vi", "Nhãn", "Giá trị", "Số tầng", "Tổng tầng", "Tỷ lệ"], rows,
          "Mẫu số gồm cả ô thiếu. Modic là loại, không phải thang mức độ. Pfirrmann giữ nguyên 1–5, không gộp lớp hiếm.", [5, 3, 4])
    rows = []
    binary = [k for k, domain in DOMAINS.items() if domain == [0, 1]]
    for col in binary:
        states = grades[col].groupby(df.patient_id).agg(lambda s: "Có ít nhất 1 tầng = 1" if s.eq(1).any() else ("Tất cả tầng = 0" if s.eq(0).all() else "Chưa xác định"))
        for state in ["Có ít nhất 1 tầng = 1", "Tất cả tầng = 0", "Chưa xác định"]:
            rows.append([col, state, int(states.eq(state).sum()), n, None])
    table("Grading BN", "Nhãn nhị phân ở cấp bệnh nhân", ["Nhãn", "Trạng thái", "Số BN", "Tổng BN", "Tỷ lệ"], rows,
          "Một tầng dương tính đủ để xếp có nhãn; chỉ xếp âm tính khi toàn bộ tầng bằng 0.", [4, 2, 3])
    rows = []
    for a, b in itertools.combinations(binary, 2):
        known = grades[a].notna() & grades[b].notna()
        rows.append([a, b, int((grades[a].eq(1) & grades[b].eq(1)).sum()), int(known.sum()), None])
    table("Grading BN", "Đồng xuất hiện nhãn tại cùng một tầng", ["Nhãn A", "Nhãn B", "Cả hai = 1", "Số tầng biết cả hai", "Tỷ lệ"], rows,
          "Thống kê mô tả, không chứng minh quan hệ nhân quả hoặc tính độc lập.", [4, 2, 3])

    paired = (p.report_vi_findings != "") & (p.report_vi_impression != "")
    rows, rare = [], []
    for fold in range(1, 6):
        col = f"fold{fold}_split"
        for split in ["train", "val", "test"]:
            selected = p[col].eq(split)
            rows.append([fold, split, int(selected.sum()), int((selected & paired).sum()), int((selected & p.report_vi_findings.ne("")).sum()), int((selected & p.report_en.ne("")).sum())])
            sel_rows = df[col].eq(split)
            for field, values in DOMAINS.items():
                for value in values:
                    rare.append([fold, split, field, value, int((sel_rows & grades[field].eq(value)).sum()), int(sel_rows.sum()), None])
    table("Chia tap", "Quy mô từng fold theo điều kiện báo cáo", ["Fold", "Split", "Toàn bộ BN", "Đủ findings + impression VI", "Có findings VI", "Có EN"], rows,
          "Các điều kiện là các tập con có thể chồng lấp. Số tầng = số BN × 5 nếu đủ cấu trúc.")
    table("Nhan theo split", "Độ phủ lớp của từng fold/split", ["Fold", "Split", "Nhãn", "Giá trị", "Số tầng", "Tổng tầng", "Tỷ lệ"], rare,
          "Có cả lớp đếm bằng 0 để thấy lớp vắng ở validation/test. Tỷ lệ dùng tổng tầng, gồm cả nhãn thiếu.", [6, 4, 5])

    rows, lengths, duplicate_rows = [], [], []
    for col in ["report_vi_technique", "report_vi_findings", "report_vi_impression", "report_en"]:
        valid = p[col].ne("")
        rows.append([col, int(valid.sum()), n, None, int((~valid).sum())])
        texts = p.loc[valid, col]
        for unit, sizes in [("ký tự", texts.str.len()), ("đơn vị tách khoảng trắng", texts.str.split().str.len())]:
            lengths.append([col, unit, len(sizes)] + [round(float(x), 1) for x in [sizes.min(), sizes.quantile(.25), sizes.median(), sizes.mean(), sizes.quantile(.75), sizes.max()]])
        norm = texts.str.replace(r"\s+", " ", regex=True).str.strip()
        counts = norm.value_counts()
        cross = []
        for fold in range(1, 6):
            temp = pd.DataFrame({"text": norm, "split": p.loc[valid, f"fold{fold}_split"]})
            cross.append(int((temp.groupby("text")["split"].nunique() > 1).sum()))
        duplicate_rows.append([col, int((counts > 1).sum()), int(counts[counts > 1].sum()), *cross])
    table("Bao cao", "Độ phủ báo cáo", ["Trường", "Có dữ liệu", "Tổng BN", "Tỷ lệ có", "Thiếu"], rows,
          "Không coi findings và impression là các bộ bệnh nhân độc lập.", [3, 1, 2])
    table("Bao cao", "Độ dài văn bản", ["Trường", "Đơn vị", "N", "Min", "Q1", "Median", "Mean", "Q3", "Max"], lengths,
          "Đếm bằng code. Đơn vị tách khoảng trắng KHÔNG phải token của LLM; không có nội dung báo cáo cá nhân trong bảng này.")
    table("Bao cao", "Văn bản lặp sau chuẩn hóa khoảng trắng", ["Trường", "Nhóm văn bản lặp", "BN thuộc nhóm lặp", "Nhóm qua split F1", "F2", "F3", "F4", "F5"], duplicate_rows,
          "Chỉ chuẩn hóa khoảng trắng, giữ hoa/thường. Văn bản giống nhau có thể do mẫu báo cáo; chưa đủ để kết luận rò rỉ bệnh nhân/ảnh.")

    checks = []
    def check(name, issues, unit, note):
        checks.append([name, int(issues), unit, note])
    check("Dòng trùng hoàn toàn (phần dư)", df.duplicated().sum(), "dòng", "Không tính dòng đầu mỗi nhóm")
    check("Khóa patient_id + level trùng (phần dư)", df.duplicated(["patient_id", "level"]).sum(), "dòng", "Khóa kỳ vọng duy nhất")
    bad_levels = df.groupby("patient_id")["level"].agg(lambda s: len(s) != 5 or set(s) != set(LEVELS))
    check("Bệnh nhân không đủ đúng 5 tầng chuẩn", bad_levels.sum(), "BN", "Kiểm cả tập tên tầng và số dòng")
    patient_fields = ["sex", "age_at_scan", "birth_year", "study_date", "sub_id", "volume", "month"] + [c for c in df if c.startswith("report_") or c.startswith("fold")]
    for col in patient_fields:
        check(f"Không nhất quán giữa 5 dòng: {col}", (df.groupby("patient_id")[col].nunique(dropna=False) > 1).sum(), "BN", "Kiểm tra trước khi lấy 1 dòng/BN")
    for col, domain in DOMAINS.items():
        check(f"Giá trị không hợp lệ: {col}", (df[col].ne("") & ~grades[col].isin(domain)).sum(), "ô", f"Miền cho phép {domain}; loại ô trống")
    for canonical, alias in [("technique", "kythuat"), ("findings", "mota"), ("impression", "ketluan")]:
        check(f"Alias VI khác nhau: {canonical}", (p[f"report_vi_{canonical}"] != p[f"report_vi_{alias}"]).sum(), "BN", "So sánh chuỗi sau strip")
    for fold in range(1, 6):
        col = f"fold{fold}_split"
        check(f"Fold {fold}: split không hợp lệ", (~p[col].isin(["train", "val", "test"])).sum(), "BN", "Miền train/val/test; kiểm split giữa 5 dòng ở trên")
    check("BN không xuất hiện đúng 1 lần ở test trong 5 fold", (p[[f"fold{i}_split" for i in range(1, 6)]].eq("test").sum(axis=1) != 1).sum(), "BN", "Không đồng nghĩa đã kiểm tra ảnh trùng giữa bệnh nhân")
    for c in ["spacing_i", "spacing_j", "spacing_k"]:
        check(f"{c} thiếu/không số/không dương", (~numeric(df[c]).gt(0)).sum(), "dòng", "Chỉ kiểm bảng, chưa kiểm header ảnh")

    json_path = root / "dataset/dataset_patients.jsonl"
    patients = [json.loads(line) for line in json_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    json_ids = [x["patient_id"] for x in patients]
    check("Patient JSONL trùng ID (phần dư)", len(json_ids) - len(set(json_ids)), "bản ghi", "Kiểm toàn bộ JSONL")
    check("ID chỉ có một phía CSV/JSONL", len(set(json_ids) ^ set(p.patient_id)), "ID", "Đối chiếu hai nguồn")
    index = df.set_index(["patient_id", "level"])
    missing_to_zero, differences, unknown_keys = 0, 0, 0
    for patient in patients:
        for level in patient["levels"]:
            key = (patient["patient_id"], level["level"])
            if key not in index.index:
                unknown_keys += 1
                continue
            record = index.loc[key]
            if isinstance(record, pd.DataFrame):
                raise ValueError("Duplicate master key: cannot compare JSON grading reliably")
            for field in DOMAINS:
                a = None if record[field] == "" else float(record[field])
                b = level["gradings"].get(field)
                differences += a != b
                missing_to_zero += a is None and b == 0
    check("Khóa tầng JSON không có trong CSV", unknown_keys, "khóa", "Đối chiếu cấu trúc")
    check("Grading CSV và JSONL khác nhau", differences, "ô", "Phân biệt null và 0")
    check("CSV thiếu nhưng JSONL = 0", missing_to_zero, "ô", "JSONL legacy mất phân biệt chưa biết/âm tính; không sửa nguồn khi EDA")
    table("Chat luong", "Kiểm tra cấu trúc và tính nhất quán", ["Kiểm tra", "Số vi phạm", "Đơn vị", "Cách hiểu"], checks,
          "0 chỉ nghĩa không thấy vi phạm theo phép kiểm này, không phải chứng nhận chất lượng lâm sàng.")
    loc = [[a, b, len(g), nr, None] for (a, b), g in df.groupby(["loc_source", "loc_qc_status"], dropna=False)]
    table("Chat luong", "Nguồn và trạng thái localization", ["Nguồn", "QC", "Số tầng", "Tổng tầng", "Tỷ lệ"], loc,
          "auto_ok là trạng thái pipeline, không đồng nghĩa đã được bác sĩ xác nhận.", [4, 2, 3])

    audit = annotation_audit(root, output / "private_annotation_audit")
    paired_n = audit["with_report"]
    rows = [["Ít nhất một cờ ở cấp BN", audit["patient_any_conflict"], paired_n, None],
            ["Cờ BN, bỏ disc_narrowing", audit["patient_conflict_without_height"], paired_n, None],
            ["Báo cáo dịch chuyển nhưng grading bulging/herniation đều 0", audit["broad_absence_conflict"], paired_n, None]]
    rows += [[f"Cờ BN: {k}", v["mismatch_patients"], paired_n, None] for k, v in audit["features"].items()]
    rows += [[k, v, paired_n, None] for k, v in sorted(audit["review_queue"].items())]
    table("Doi chieu", "Cờ sàng lọc grading – báo cáo", ["Tiêu chí", "Số BN", "BN có báo cáo được xét", "Tỷ lệ"], rows,
          "Heuristic chưa được thẩm định lâm sàng. A/B/C phân nhóm theo tiêu chí rộng hơn 'cờ cấp BN'; A+B không phải tập con của 131. Không nêu ≠ âm tính. 9 BN không có báo cáo không được đánh giá.", [3, 1, 2])
    matrix = [[feature, state, count, paired_n, None] for feature, detail in audit["features"].items() for state, count in detail["matrix"].items()]
    table("Doi chieu", "Ma trận trạng thái grading / báo cáo", ["Nhãn", "Grading / report", "Số BN", "Tổng BN được xét", "Tỷ lệ"], matrix,
          "unspecified = không phát hiện phát biểu; internal_conflict = parser tìm thấy cả khẳng định/phủ định, cần đọc lại. Không dùng bảng này tính độ chính xác bác sĩ.", [4, 2, 3])

    section_counts = {f: dict(opposite=0, omitted=0, added=0, internal=0) for f in FIELDS}
    section_n = 0
    for patient in patients:
        vi = patient["reports"]["vi"]
        findings = vi.get("findings", vi.get("mo_ta", []))
        impression = vi.get("impression", vi.get("ket_luan", []))
        if not findings or not impression:
            continue
        section_n += 1
        fe, ie = extract(findings, "findings"), extract(impression, "impression")
        for feature in FIELDS:
            a, b = report_state(fe, feature), report_state(ie, feature)
            section_counts[feature]["opposite"] += (a, b) in [("positive", "negative"), ("negative", "positive")]
            section_counts[feature]["omitted"] += a == "positive" and b == "unspecified"
            section_counts[feature]["added"] += a == "unspecified" and b == "positive"
            section_counts[feature]["internal"] += "internal_conflict" in (a, b)
    table("Doi chieu", "Sàng lọc riêng findings – impression", ["Nhãn", "Trái dấu rõ theo parser", "Findings dương / impression không nêu", "Findings không nêu / impression dương", "Ít nhất 1 phần có cờ nội bộ", "BN có cả 2 phần"],
          [[f, *v.values(), section_n] for f, v in section_counts.items()],
          "Chỉ 6 nhóm thuật ngữ, xét cấp BN, chưa kiểm toàn bộ vị trí/mức độ. Không nêu trong impression có thể là tóm tắt hợp lệ, KHÔNG tự coi là lỗi. Trái dấu vẫn cần bác sĩ đọc xác nhận; 0 không chứng minh không có mâu thuẫn.")

    inventory = []
    paths = [source, json_path, root / "scripts/audit_annotation_consistency.py", Path(__file__).resolve()]
    for directory in ["dataset_local/grading", "dataset_local/localize", "dataset_local/folds", "dataset_local/reports_json", "dataset_local/reports_text", "dataset_local/reports_text_v1_reference", "dataset/sft_data", "dataset_local/nifti", "dataset_local/dicom"]:
        folder = root / directory
        files = sorted(x for x in folder.rglob("*") if x.is_file()) if folder.exists() else []
        inventory.append([directory, "Có" if folder.exists() else "Không có local", len(files), sum(x.stat().st_size for x in files)])
        # No patient filename or text in aggregate output; content fingerprint is enough.
        if files:
            digest = hashlib.sha256()
            for f in files:
                digest.update(str(f.relative_to(folder)).replace("\\", "/").encode())
                digest.update(hashlib.sha256(f.read_bytes()).digest())
            inventory[-1].append(digest.hexdigest())
        else:
            inventory[-1].append("")
    table("Nguon va pham vi", "Danh mục nguồn local", ["Thư mục", "Hiện diện", "Số file", "Bytes", "SHA256 tập file"], inventory,
          "Đếm file gồm cả README. Hash tập file = SHA256 chuỗi tên tương đối + SHA256 nội dung từng file theo thứ tự; không công khai tên file bệnh nhân.")
    table("Nguon va pham vi", "Dấu vân tay đầu vào và code", ["Nguồn", "Bytes", "SHA256"], [[str(x.relative_to(root)).replace("\\", "/"), x.stat().st_size, hashlib.sha256(x.read_bytes()).hexdigest()] for x in paths])
    sft = []
    for path in sorted((root / "dataset/sft_data").glob("*.jsonl")):
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        sft.append([path.name, len(data), "bản ghi SFT; không mặc định là số BN duy nhất"])
    table("Nguon va pham vi", "Các file SFT legacy", ["File", "Số bản ghi", "Đơn vị / lưu ý"], sft,
          "EDA kiểm số dòng JSON hợp lệ; không suy ra tập này là đầu vào chuẩn V2 hoặc đủ findings + impression.")

    # Reconciliation guards catch denominator/grain mistakes before export.
    assert sum(x[1] for x in audit["review_queue"].items()) == paired_n
    assert all(sum(v["matrix"].values()) == paired_n for v in audit["features"].values())
    for col, domain in DOMAINS.items():
        assert grades[col].isin(domain).sum() + grades[col].isna().sum() + (grades[col].notna() & ~grades[col].isin(domain)).sum() == nr
    for t in tables:
        if t["ratio"]:
            dest, numerator, denominator = t["ratio"]
            for row in t["rows"]:
                row[dest] = row[numerator] / row[denominator] if row[denominator] else None
        pd.DataFrame(t["rows"], columns=t["headers"]).to_csv(output / f"table_{tables.index(t)+1:02d}.csv", index=False, encoding="utf-8-sig")
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(), "tables": tables,
               "scope": "Aggregate EDA of local tabular/text annotations, not MRI pixel EDA or clinical adjudication"}
    (output / "eda_tables.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    md = ["# EDA toàn bộ bảng dữ liệu hiện có", "", f"Tạo tự động: {payload['generated_utc']}", "",
          "Các số liệu dưới đây được tính bằng code trên toàn bộ dữ liệu local, không phải đọc từng dòng rồi ước lượng. Diễn giải dựa trên schema và quy tắc audit. Không có GPU hay model được chạy.", "",
          "## Phạm vi và giới hạn", "",
          "- Một dòng master là một tầng đĩa đệm; nhân khẩu học và báo cáo được khử lặp theo patient_id.",
          "- Bao gồm missingness mọi cột, phân bố nhãn, tầng, fold, nhân khẩu, độ dài/lặp văn bản, localization, cấu trúc và cờ bất nhất.",
          "- Không kiểm pixel ảnh, chất lượng chuỗi MRI, ảnh trùng, orientation hoặc tính đúng của chẩn đoán; ảnh không có local. Không coi cờ heuristic là tỷ lệ gán nhãn sai.",
          f"- Cờ cấp bệnh nhân: {audit['patient_any_conflict']}/{paired_n} ({audit['patient_any_conflict']/paired_n:.1%}) ca có báo cáo được xét. A/B/C theo tiêu chí khác, không phải các phần của số ca có cờ này. Không phải tỷ lệ sai nhãn đã xác nhận.",
          "- Null/ô trống không được đổi thành 0. Tuổi và ngày theo dữ liệu đã khử định danh; không suy luận dịch tễ đại diện dân số.",
          "- Tỷ lệ chỉ là mô tả mẫu nghiên cứu, không có kiểm định thống kê hay suy luận quần thể.", "",
          "## Tái lập", "", "```powershell", "python scripts/build_complete_eda.py --output outputs/eda_2026-09-18", "```", "",
          "Cần pandas. Script chỉ đọc nguồn, xuất CSV tổng hợp và JSON; audit theo bệnh nhân nằm trong output bị gitignore. File Excel trình bày lại chính các bảng tổng hợp này.", ""]
    for t in tables:
        md += [f"## {t['title']}", "", t["note"], "", "| " + " | ".join(t["headers"]) + " |", "| " + " | ".join(["---"] * len(t["headers"])) + " |"]
        for row in t["rows"]:
            cells = []
            for i, x in enumerate(row):
                text = "—" if x is None else (f"{100*x:.1f}%" if t["ratio"] and i == t["ratio"][0] else str(x))
                cells.append(text.replace("|", "\\|").replace("\n", " "))
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    (output / "EDA_complete.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"patients": n, "rows": nr, "tables": len(tables), "sheets": len(set(t['sheet'] for t in tables)), "json_missing_to_zero": missing_to_zero, "audit_evaluated": paired_n}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/eda_2026-09-18"))
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve())
