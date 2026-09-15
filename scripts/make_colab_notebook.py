"""
Script: make_colab_notebook.py
Tạo tệp Jupyter Notebook cho Google Colab: notebooks/V1_End_to_End_VLM_Training.ipynb
"""

import sys
import json
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏥 V1: Huấn luyện End-to-End VLM Sinh Báo Cáo MRI Cột Sống\n",
                "### Đề tài: Calibrated Level-wise Spine MRI Report Generation via Medical VLM & LoRA\n",
                "\n",
                "Notebook này thực hiện huấn luyện mô hình cơ sở **V1 (End-to-End Vision-Language Model)** trên **Google Colab (GPU T4 / A100)**.\n",
                "- **Mã nguồn:** Tải trực tiếp từ [GitHub: MRI-report-generator](https://github.com/kttt294/MRI-report-generator)\n",
                "- **Dữ liệu:** Liên kết trực tiếp từ Google Drive của bộ dữ liệu PSPINES.\n",
                "- **Mô hình:** Qwen2.5-VL-3B-Instruct kết hợp **4-bit QLoRA** tiết kiệm VRAM."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Bước 1: Kiểm tra phần cứng GPU"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Bước 2: Tải mã nguồn từ GitHub"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import os\n",
                "%cd /content\n",
                "if not os.path.exists('MRI-report-generator'):\n",
                "    !git clone https://github.com/kttt294/MRI-report-generator.git\n",
                "%cd /content/MRI-report-generator\n",
                "!git pull"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Bước 3: Cài đặt các thư viện phụ thuộc"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "!pip install -q -r requirements.txt\n",
                "!pip install -q nibabel qwen-vl-utils torchvision pyyaml"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Bước 4: Kết nối Google Drive và Trỏ Dữ liệu\n",
                "> **HƯỚNG DẪN:** Thư mục dữ liệu được chia sẻ từ liên kết:\n",
                "> `https://drive.google.com/drive/u/1/folders/1uJuVbDIoNQP4Qrva9ZTgZSePtGfHj-Bg`\n",
                "> \n",
                "> Trên Google Drive của bạn:\n",
                "> 1. Vào mục **'Được chia sẻ với tôi' (Shared with me)**.\n",
                "> 2. Nhấp chuột phải vào thư mục `dataset` -> Chọn **'Thêm lối tắt vào Drive' (Add shortcut to Drive)** -> Đặt vào **'Drive của tôi' (My Drive)**.\n",
                "> 3. Khi đó đường dẫn trong Colab sẽ xuất hiện tại `/content/drive/MyDrive/dataset`."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "# Kiểm tra thư mục dataset trên Google Drive\n",
                "DRIVE_DATASET = Path('/content/drive/MyDrive/dataset')\n",
                "\n",
                "if DRIVE_DATASET.exists():\n",
                "    print('-> Đã tìm thấy thư mục dataset trên Google Drive:', [p.name for p in DRIVE_DATASET.iterdir()][:6])\n",
                "    if not os.path.exists('dataset_local'):\n",
                "        !ln -s /content/drive/MyDrive/dataset dataset_local\n",
                "        print('-> Đã tạo liên kết dataset_local -> Google Drive dataset!')\n",
                "else:\n",
                "    print('CHƯA TÌM THẤY! Vui lòng kiểm tra bạn đã tạo Lối tắt (Shortcut) thư mục dataset vào MyDrive chưa.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Bước 5: Chạy Script Tổng hợp Dữ liệu vào `dataset/`"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "!python scripts/consolidate_dataset.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Bước 6: Khởi động Huấn luyện V1 End-to-End VLM bằng LoRA\n",
                "Huấn luyện mô hình Qwen2.5-VL-3B-Instruct bằng 4-bit QLoRA trên GPU T4."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "!python src/train_v1.py \\\n",
                "    --config configs/v1_config.yaml \\\n",
                "    --master_csv dataset/dataset_master.csv \\\n",
                "    --nifti_dir dataset_local/nifti \\\n",
                "    --output_dir ./checkpoints/v1_vlm_lora \\\n",
                "    --epochs 5"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Bước 7: Kiểm thử Suy luận (Inference Test) trên 1 ca bệnh"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import torch\n",
                "from peft import PeftModel\n",
                "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration\n",
                "from src.data.v1_dataset import SpineVLMDataset\n",
                "\n",
                "adapter_path = './checkpoints/v1_vlm_lora/final_adapter'\n",
                "base_model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'\n",
                "\n",
                "print('Đang nạp mô hình sau huấn luyện để kiểm thử...')\n",
                "processor = AutoProcessor.from_pretrained(adapter_path)\n",
                "base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n",
                "    base_model_id,\n",
                "    torch_dtype=torch.bfloat16,\n",
                "    device_map='auto'\n",
                ")\n",
                "model = PeftModel.from_pretrained(base_model, adapter_path)\n",
                "model.eval()\n",
                "\n",
                "# Nạp tập test\n",
                "test_dataset = SpineVLMDataset(\n",
                "    master_csv='dataset/dataset_master.csv',\n",
                "    nifti_dir='dataset_local/nifti',\n",
                "    split='test',\n",
                "    fold=1,\n",
                "    language='vi'\n",
                ")\n",
                "\n",
                "sample = test_dataset[0]\n",
                "print(f\"\\n=== BỆNH NHÂN KIỂM THỬ: {sample['patient_id']} ===\")\n",
                "\n",
                "conv = [\n",
                "    {\n",
                "        'role': 'user',\n",
                "        'content': [\n",
                "            {'type': 'image'},\n",
                "            {'type': 'text', 'text': sample['prompt']}\n",
                "        ]\n",
                "    }\n",
                "]\n",
                "text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)\n",
                "inputs = processor(text=[text], images=[sample['image']], return_tensors='pt').to('cuda')\n",
                "\n",
                "with torch.no_grad():\n",
                "    generated_ids = model.generate(**inputs, max_new_tokens=300)\n",
                "    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]\n",
                "    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]\n",
                "\n",
                "print('\\n--- BÁO CÁO VLM SINH RA ---')\n",
                "print(output_text)\n",
                "print('\\n--- BÁO CÁO CHUẨN CỦA BÁC SĨ (GROUND TRUTH) ---')\n",
                "print(sample['target_text'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Bước 8: Lưu Adapter LoRA về Google Drive\n",
                "Lưu trữ vĩnh viễn trọng số adapter (~30MB) về Google Drive để không bị mất khi hết phiên Colab."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "!mkdir -p /content/drive/MyDrive/Spine_MRI_Checkpoints\n",
                "!cp -r ./checkpoints/v1_vlm_lora/final_adapter /content/drive/MyDrive/Spine_MRI_Checkpoints/v1_adapter\n",
                "print('ĐÃ LƯU ADAPTER THÀNH CÔNG VÀO /content/drive/MyDrive/Spine_MRI_Checkpoints/v1_adapter!')"
            ]
        }
    ]
}

notebook_path = Path("notebooks/V1_End_to_End_VLM_Training.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"-> Đã tạo thành công Notebook: {notebook_path}")
