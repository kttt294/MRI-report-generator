"""Generate thin Kaggle launchers. All implementation stays in versioned .py files."""
from pathlib import Path
import nbformat as nbf


def notebook(task):
    name = {"v1-train": "V1 MRI → báo cáo", "v2-generate": "V2 JSON → báo cáo",
            "v2-train": "V2 LoRA — chỉ target đã duyệt"}[task]
    configuration = f'''CONFIG = {{
    "task": "{task}", "mode": "smoke", "run_name": "{task}-smoke-01",
    "annotations_root": "/kaggle/input/CHANGE-ME-annotations",
    "annotations_dataset": "OWNER/SLUG/VERSION",
    "images_root": "/kaggle/input/CHANGE-ME-images/nifti",
    "images_dataset": "OWNER/SLUG/VERSION",
    "work_root": "/kaggle/working", "cache_root": "/tmp/mri-cache",
    "fold": 1, "gpu_index": "0", "max_runtime_minutes": 30,
    "resume_from": None,
    "reviewed_targets": None,
    "report_overrides": {{"backend": "template"}},
}}'''
    cells = [nbf.v4.new_markdown_cell(f"# {name}\n\n"
        "Notebook private. Xem `docs/README_KAGGLE.md` trong repo trước khi chạy. "
        "Add Input các dataset private; sao chép đúng đường dẫn và version vào cell tham số. "
        "V2 template chạy CPU, V1/V2 LoRA cần GPU. Không dùng output template làm nhãn bác sĩ.\n\n"
        "Chạy smoke trước; đổi `mode` thành `full` và đặt `run_name` mới cho run chính. "
        "Sau khi dừng, xác nhận checkpoint nằm trong Output của Saved Version."),
        nbf.v4.new_code_cell('CODE_REF = "codex/kaggle-v1-v2"  # đổi thành commit SHA để khóa thí nghiệm\n'
                             'REPO_URL = "https://github.com/kttt294/MRI-report-generator.git"'),
        nbf.v4.new_code_cell('''import subprocess, sys
from pathlib import Path
REPO = Path("/kaggle/working/repo")
if REPO.exists():
    raise RuntimeError("Repo đã tồn tại: restart session sạch hoặc dùng checkout hiện tại có kiểm soát.")
subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
subprocess.run(["git", "checkout", CODE_REF], cwd=REPO, check=True)
CODE_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
print("Code commit:", CODE_SHA)'''),
        nbf.v4.new_code_cell('''subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-kaggle.txt"], cwd=REPO, check=True)
subprocess.run([sys.executable, "scripts/check_environment.py"], cwd=REPO, check=True)'''),
        nbf.v4.new_code_cell(configuration),
        nbf.v4.new_code_cell('''import json
CONFIG["code_commit"] = CODE_SHA
config_path = Path("/kaggle/working/cloud_run.json")
config_path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
subprocess.run([sys.executable, "scripts/cloud_run.py", "--config", str(config_path)], cwd=REPO, check=True)'''),
        nbf.v4.new_code_cell('''run_dir = Path(CONFIG["work_root"]) / "runs" / CONFIG["run_name"]
print("Kết quả:", run_dir)
print("\\n".join(str(p.relative_to(run_dir)) for p in sorted(run_dir.iterdir())))
print("Save Version → kiểm tra Output; final_adapter để inference, checkpoint-* có COMPLETE.json để resume.")'''),
        nbf.v4.new_markdown_cell("## Chạy tiếp / inference\n\n"
        "V1: sau train, tạo run mới với `task='v1-infer'`, thêm `adapter_path` trỏ đến "
        "`final_adapter`, và `split='val'` khi đang phát triển. Chỉ dùng test sau khi khóa thiết kế.\n\n"
        "Resume: Add Input outputs của Saved Version cũ, đặt `resume_from` tới `checkpoint-N` "
        "có `COMPLETE.json`; giữ code SHA, model/config, fold, mode và data versions như cũ. "
        "Đổi `run_name`, giữ thời gian dự phòng để lưu outputs. Smoke không được resume thành full.\n\n"
        "V2 LLM: đặt `report_overrides` theo README; kiểm tra cả attempts và fallback. "
        "V2 train cần file targets có duyệt thật, không đổi `pending` hàng loạt thành `accepted`.")]
    for i, cell in enumerate(cells):
        cell.id = f"{task}-{i:02d}"
    result = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    nbf.validate(result)
    return result


def main():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    root.mkdir(exist_ok=True)
    for task, filename in (("v1-train", "Kaggle_V1_Training.ipynb"),
                           ("v2-generate", "Kaggle_V2_Report.ipynb"),
                           ("v2-train", "Kaggle_V2_Training.ipynb")):
        nbf.write(notebook(task), root / filename)
    # Replace the executable legacy notebook; keep its discoverable filename.
    old = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        "# Notebook cũ đã ngừng sử dụng\n\nDùng [Kaggle_V1_Training.ipynb](Kaggle_V1_Training.ipynb) "
        "và [hướng dẫn](../docs/README_KAGGLE.md). Toàn bộ training nằm trong module `.py`. "
        "Không chạy vòng lặp cũ vì masking và đường đọc ảnh đã thay đổi.")])
    old.cells[0].id = "legacy-migration"
    nbf.write(old, root / "V1_End_to_End_VLM_Training.ipynb")


if __name__ == "__main__":
    main()
