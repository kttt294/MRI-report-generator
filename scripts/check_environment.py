"""Fail early on a broken torch/torchvision pair, without replacing Kaggle CUDA."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import environment_receipt


def check():
    import torch
    import torchvision
    from transformers import Trainer, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "total_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)}
               for i in range(torch.cuda.device_count())]
    return {**environment_receipt(), "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda, "devices": devices}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
