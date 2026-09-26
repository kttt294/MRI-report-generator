import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Fine-tune Qwen3-VL-4B-Instruct trên báo cáo MRI cột sống thắt lưng (Fold 1)\n",
            "\n",
            "Thử nghiệm này khắc phục triệt để các hạn chế của đợt fine-tune V2-2 trước đây:\n",
            "- **Model nền:** `Qwen/Qwen3-VL-4B-Instruct` (đã chứng minh khả năng tuân thủ JSON 100% và không bị lặp vô tận).\n",
            "- **Tăng số bước huấn luyện:** 8 epochs (~272 gradient steps so với chỉ 34 steps ở V2-2 cũ).\n",
            "- **Tối ưu siêu tham số:** Learning rate 1.5e-4 với Cosine decay và warmup 5%.\n",
            "- **Decoding Guard khi test:** Bật `repetition_penalty=1.15` và `no_repeat_ngram_size=5` trên 47 ca test.\n",
            "- **Đánh giá tự động:** Đánh giá ngay trên tập test và đối chiếu với bản trước."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import os, sys, subprocess, time\n",
            "CODE_SHA = \"83a43e4\"\n",
            "REPO = Path(\"/kaggle/working/repo\")\n",
            "RUN_DIR = Path(\"/kaggle/working/runs/qwen3-finetune-01\")\n",
            "os.environ[\"HF_HOME\"] = \"/tmp/mri-qwen3-hf\"\n",
            "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"0\"\n",
            "os.environ[\"TOKENIZERS_PARALLELISM\"] = \"false\"\n",
            "assert not REPO.exists(), \"Dùng phiên Kaggle mới để giữ môi trường sạch sẽ\"\n",
            "setup_start = time.monotonic()\n",
            "subprocess.run([\"git\", \"clone\", \"https://github.com/kttt294/MRI-report-generator.git\", str(REPO)], check=True)\n",
            "subprocess.run([\"git\", \"checkout\", CODE_SHA], cwd=REPO, check=True)\n",
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", str(REPO / \"requirements-qwen3-evaluation.txt\"), \"peft>=0.14.0\"], check=True)\n",
            "print(\"Code commit:\", CODE_SHA, \"Setup seconds:\", round(time.monotonic() - setup_start, 1))\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "assert torch.cuda.is_available(), \"Bật GPU T4 trong Kaggle Settings!\"\n",
            "print(\"GPU:\", torch.cuda.get_device_name(0))\n",
            "print(\"VRAM:\", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), \"GB\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, sys\n",
            "print(\"=== BẮT ĐẦU HUẤN LUYỆN VÀ ĐÁNH GIÁ QWEN3-VL-4B ===\")\n",
            "subprocess.run([sys.executable, \"-u\", \"scripts/train_qwen3_finetune.py\",\n",
            "                \"--input-root\", \"/kaggle/input\",\n",
            "                \"--config\", \"configs/qwen3_finetune.yaml\",\n",
            "                \"--output\", str(RUN_DIR)], cwd=REPO, check=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json, pandas as pd\n",
            "from IPython.display import display\n",
            "metrics_path = RUN_DIR / \"metrics.json\"\n",
            "if metrics_path.exists():\n",
            "    with open(metrics_path, encoding=\"utf-8\") as f:\n",
            "        metrics = json.load(f)\n",
            "    print(\"\\n--- KẾT QUẢ CUỐI CÙNG ---\")\n",
            "    for k, v in metrics.items():\n",
            "        print(f\"{k}: {v}\")\n",
            "print(\"File human review để bác sĩ đọc đối chiếu:\", RUN_DIR / \"human_review_qwen3_finetuned.csv\")\n",
            "print(\"Adapter đã lưu tại:\", RUN_DIR / \"best_adapter\")\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

Path("notebooks/Kaggle_V2_2_Qwen3_Finetuning.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Successfully created notebooks/Kaggle_V2_2_Qwen3_Finetuning.ipynb")
