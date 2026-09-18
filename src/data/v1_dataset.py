"""Patient-level MRI dataset with strict splits, target aliases and real images."""
import csv
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset
from src.contracts.report_input import LEVELS
from src.data.report_targets import clean_text, vi_sections, format_target
from src.data.mri_images import cached_views, resolve_volume, volume_index


class SpineVLMDataset(Dataset):
    def __init__(self, master_csv, nifti_dir, split="train", fold=1, language="vi",
                 image_cache_dir=None, target_size=(384, 384), processor=None,
                 offsets=(0,), slice_selection="middle", require_both_sections=True,
                 require_images=True):
        if split not in {"train", "val", "test"} or fold not in range(1, 6):
            raise ValueError("Invalid split/fold")
        if language not in {"vi", "en"} or slice_selection not in {"middle", "annotation_center"}:
            raise ValueError("Invalid language/slice selection")
        self.master_csv, self.nifti_dir = Path(master_csv), Path(nifti_dir)
        self.split, self.fold, self.language = split, fold, language
        self.image_cache_dir, self.target_size = image_cache_dir, tuple(target_size)
        self.offsets, self.slice_selection = tuple(offsets), slice_selection
        self.samples, self.excluded = [], []
        groups = defaultdict(list)
        with self.master_csv.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                groups[row["patient_id"]].append(row)
        index = volume_index(self.nifti_dir) if require_images else None
        for pid, rows in sorted(groups.items()):
            if len(rows) != 5 or {r["level"] for r in rows} != set(LEVELS):
                raise ValueError(f"Patient {pid}: expected five unique levels")
            for fold_number in range(1, 6):
                values = {r.get(f"fold{fold_number}_split") for r in rows}
                if len(values) != 1 or not values <= {"train", "val", "test"}:
                    raise ValueError(f"Patient {pid}: inconsistent fold {fold_number}")
            if rows[0][f"fold{fold}_split"] != split:
                continue
            targets = {vi_sections(r, csv=True) if language == "vi" else (clean_text(r.get("report_en")), "") for r in rows}
            if len(targets) != 1:
                raise ValueError(f"Patient {pid}: inconsistent report across levels")
            findings, impression = targets.pop()
            if not findings or (language == "vi" and require_both_sections and not impression):
                self.excluded.append({"patient_id": pid, "reason": "missing_target_section"})
                continue
            volumes = {r.get("volume", "").strip() for r in rows}
            if len(volumes) != 1 or not next(iter(volumes)):
                raise ValueError(f"Patient {pid}: one explicit source volume is required")
            volume = volumes.pop()
            path = resolve_volume(self.nifti_dir, volume, index) if require_images else None
            center = None
            if slice_selection == "annotation_center":
                center = [sum(float(r[k]) for r in rows) / 5 for k in ("voxel_i", "voxel_j", "voxel_k")]
            prompt = ("Dựa vào các ảnh MRI cột sống thắt lưng được cung cấp, viết [MÔ TẢ] và [KẾT LUẬN]. "
                      "Chỉ mô tả những gì đánh giá được trên ảnh; không tự bổ sung thông tin chưa thấy.") if language == "vi" else (
                      "Describe the supplied lumbar MRI images and provide findings and impression. "
                      "Do not invent findings not supported by the supplied views.")
            self.samples.append({"patient_id": pid, "volume": volume, "path": str(path) if path else None,
                "source_center": center, "prompt": prompt,
                "target_text": format_target(findings, impression) if language == "vi" else findings})
        if not self.samples:
            raise ValueError(f"No eligible {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        if item["path"] is None:
            raise RuntimeError("This dataset was loaded for metadata inspection only")
        images = cached_views(item["path"], self.image_cache_dir, self.target_size,
                              self.offsets, item["source_center"])
        user = {"role": "user", "content": [{"type": "image"} for _ in images] +
                [{"type": "text", "text": item["prompt"]}]}
        assistant = {"role": "assistant", "content": [{"type": "text", "text": item["target_text"]}]}
        return {**item, "image": images[0], "images": images, "conversation": [user, assistant]}

    def manifest(self):
        return {"split": self.split, "fold": self.fold, "count": len(self), "excluded": self.excluded,
                "slice_selection": self.slice_selection, "offsets": self.offsets,
                "samples": [{k: r[k] for k in ("patient_id", "volume", "path", "source_center")} for r in self.samples]}
