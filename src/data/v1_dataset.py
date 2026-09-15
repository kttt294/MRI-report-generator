"""
Module: src.data.v1_dataset
Mục đích: Dataset loader cho V1 End-to-End VLM.
Tự động trích xuất lát cắt Sagittal T2 chính giữa từ file NIfTI (.nii.gz),
chuẩn hoá độ tương phản và ghép cặp với văn bản báo cáo y tế (Việt/Anh).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError:
    nib = None


def normalize_slice_to_pil(slice_2d: np.ndarray, target_size: Tuple[int, int] = (384, 384)) -> Image.Image:
    """
    Chuẩn hoá lát cắt cường độ tín hiệu MRI sang ảnh RGB PIL Image (8-bit [0, 255])
    áp dụng windowing phân vị [p1, p99] để làm nổi bật đĩa đệm và bao màng cứng.
    """
    # Xoay và lật để hướng đúng giải phẫu chuẩn y khoa (đầu lên trên, lưng bên phải)
    slice_2d = np.rot90(slice_2d)
    
    # Lọc ngoại lai cường độ
    p_low, p_high = np.percentile(slice_2d, (1, 99))
    clipped = np.clip(slice_2d, p_low, p_high)
    
    # Chuẩn hoá về [0, 255]
    if p_high > p_low:
        normalized = ((clipped - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
    else:
        normalized = np.zeros_like(clipped, dtype=np.uint8)
    
    img = Image.fromarray(normalized)
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        
    # Chuyển sang 3 kênh RGB (chuẩn của các bộ mã hoá Vision Transformer)
    return img.convert("RGB")


def extract_key_sagittal_slice(
    nii_path: Union[str, Path],
    target_k: Optional[float] = None,
    target_size: Tuple[int, int] = (384, 384)
) -> Image.Image:
    """
    Trích xuất lát cắt đứng dọc chính giữa (Mid-sagittal) từ file NIfTI 3D.
    """
    if nib is None:
        raise ImportError("Vui lòng cài đặt nibabel: pip install nibabel")
        
    nimg = nib.load(str(nii_path))
    vol = nimg.get_fdata()
    num_slices = vol.shape[2]
    
    if target_k is not None and not np.isnan(target_k):
        slice_idx = int(np.clip(np.round(target_k), 0, num_slices - 1))
    else:
        slice_idx = num_slices // 2
        
    slice_2d = vol[:, :, slice_idx]
    return normalize_slice_to_pil(slice_2d, target_size=target_size)


class SpineVLMDataset(Dataset):
    """
    PyTorch Dataset cho bài toán V1 End-to-End VLM Spine Report Generation.
    """
    def __init__(
        self,
        master_csv: Union[str, Path],
        nifti_dir: Union[str, Path],
        split: str = "train",
        fold: int = 1,
        language: str = "vi",
        image_cache_dir: Optional[Union[str, Path]] = None,
        target_size: Tuple[int, int] = (384, 384),
        processor = None
    ):
        super().__init__()
        self.master_csv = Path(master_csv)
        self.nifti_dir = Path(nifti_dir)
        self.split = split
        self.fold = fold
        self.language = language
        self.target_size = target_size
        self.processor = processor
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None
        if self.image_cache_dir:
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)
            
        self._prepare_data()

    def _prepare_data(self):
        df = pd.read_csv(self.master_csv)
        split_col = f"fold{self.fold}_split"
        
        if split_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột phân chia fold '{split_col}' trong {self.master_csv}")
            
        # Lọc theo split ('train', 'val', 'test')
        df_split = df[df[split_col] == self.split].copy()
        
        # Gom nhóm theo từng bệnh nhân
        self.samples = []
        for pid, group in df_split.groupby("patient_id"):
            first_row = group.iloc[0]
            
            # Kiểm tra báo cáo mục tiêu
            if self.language == "vi":
                mota = str(first_row.get("report_vi_mota", "")).strip()
                ketluan = str(first_row.get("report_vi_ketluan", "")).strip()
                if not mota and not ketluan:
                    continue
                target_text = f"[MÔ TẢ]:\n{mota}\n\n[KẾT LUẬN]:\n{ketluan}"
                prompt_text = (
                    "Dựa trên hình ảnh cộng hưởng từ (MRI) cột sống thắt lưng chuỗi xung Sagittal T2, "
                    "hãy quan sát toàn bộ các tầng đĩa đệm từ L1/L2 đến L5/S1, ống sống và rễ thần kinh "
                    "để viết báo cáo chẩn đoán lâm sàng gồm phần [MÔ TẢ] và [KẾT LUẬN]."
                )
            else:
                target_text = str(first_row.get("report_en", "")).strip()
                if not target_text:
                    continue
                prompt_text = (
                    "Based on the provided Sagittal T2 lumbar spine MRI scan, examine all intervertebral "
                    "disc levels from L1/L2 to L5/S1, spinal canal, and neural foramina, then generate the "
                    "complete diagnostic report and impression."
                )
                
            # Tính chỉ số lát cắt chính giữa
            k_mean = group["voxel_k"].mean()
            
            self.samples.append({
                "patient_id": str(pid),
                "target_k": k_mean,
                "prompt": prompt_text,
                "target_text": target_text
            })
            
        print(f"[{self.split.upper()} - Fold {self.fold}] Tải thành công {len(self.samples)} ca bệnh ({self.language}).")

    def __len__(self) -> int:
        return len(self.samples)

    def get_image(self, patient_id: str, target_k: float) -> Image.Image:
        """Nạp ảnh từ cache hoặc trích xuất trực tiếp từ file NIfTI"""
        # 1. Kiểm tra cache
        if self.image_cache_dir:
            cache_path = self.image_cache_dir / f"{patient_id}_sag_t2.png"
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")
                
        # 2. Trích xuất từ file NIfTI
        nii_name = f"{patient_id}_sag_t2.nii.gz"
        nii_path = self.nifti_dir / nii_name
        
        if nii_path.exists():
            img = extract_key_sagittal_slice(nii_path, target_k=target_k, target_size=self.target_size)
            if self.image_cache_dir:
                img.save(self.image_cache_dir / f"{patient_id}_sag_t2.png")
            return img
            
        # 3. Fallback: Nếu chưa có file NIfTI, tạo ảnh placeholder (cho phép test pipeline)
        dummy = Image.new("RGB", self.target_size, color=(30, 30, 30))
        return dummy

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = self.get_image(sample["patient_id"], sample["target_k"])
        
        # Định dạng tin nhắn chuẩn hội thoại của các mô hình VLM hiện đại
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["prompt"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["target_text"]}
                ]
            }
        ]
        
        return {
            "patient_id": sample["patient_id"],
            "image": image,
            "conversation": conversation,
            "prompt": sample["prompt"],
            "target_text": sample["target_text"]
        }
