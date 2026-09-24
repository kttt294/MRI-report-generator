"""Prepare private annotations, verify splits, and train V2-2 on Kaggle."""
import argparse
import os
import sys
from pathlib import Path

# Kaggle T4 x2: QLoRA model is intentionally placed on one GPU. Set this
# before importing any module that may initialize torch/CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from scripts.cloud_prepare import prepare
from scripts.v2_2_preflight import preflight
from src.train_v2_2 import train
from src.io_utils import write_json


def run(annotations_root, output, mode, config, runtime_minutes):
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    derived = output / "derived"
    prepared = prepare(annotations_root, derived, source_versions={"annotations": "lumbar-mri-annotations/1"})
    patients = derived / "dataset/dataset_patients.jsonl"
    master = derived / "dataset/dataset_master.csv"
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    receipt = preflight(patients, master, int(cfg["fold"]), output / "preflight.json",
        cfg["model"]["name_or_path"], int(cfg["max_sequence_length"]))
    write_json(output / "cloud_receipt.json", {"mode": mode, "prepared": prepared,
        "preflight_counts": receipt["counts"]})
    return train(cfg, patients, master, output / "training", smoke=mode == "smoke",
                 max_runtime_minutes=runtime_minutes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--config", default="configs/v2_2_train.yaml")
    p.add_argument("--runtime-minutes", type=float, default=180)
    a = p.parse_args()
    print(run(a.annotations_root, a.output, a.mode, a.config, a.runtime_minutes))
