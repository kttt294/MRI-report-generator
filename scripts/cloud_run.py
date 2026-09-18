"""Thin cloud orchestrator; sets the single GPU before importing torch."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.io_utils import content_hash, strict_loads, write_json, environment_receipt


def merge_overrides(base, overrides):
    for key, value in overrides.items():
        if key not in base: raise ValueError(f"Unknown config override: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict): raise ValueError(f"Expected mapping for {key}")
            merge_overrides(base[key], value)
        else:
            base[key] = value


def run(config):
    if config.get("gpu_index", "0") not in {"0", "1"}:
        raise ValueError("MVP supports one GPU: gpu_index 0 or 1")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpu_index", "0")
    os.environ["HF_HOME"] = str(Path(config.get("cache_root", "/tmp/mri-cache")) / "huggingface")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    task = config["task"]
    allowed = {"v1-train", "v1-infer", "v2-generate", "v2-train"}
    if task not in allowed or config.get("mode", "smoke") not in {"smoke", "full"}:
        raise ValueError("Invalid cloud task/mode")
    name = config["run_name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("run_name must contain letters, digits, underscores or hyphens")
    work = Path(config["work_root"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < 1024 ** 3:
        raise ValueError("Less than 1 GB free on work volume")
    run_dir = work / "runs" / name
    if run_dir.exists():
        raise ValueError("run_name already exists; use a new run_name")
    for required in ("annotations_root", "annotations_dataset"):
        if not config.get(required):
            raise ValueError(f"Set {required}; record private dataset version for reproducibility")
    if task.startswith("v1") and (not config.get("images_root") or not config.get("images_dataset")):
        raise ValueError("V1 requires images_root and images_dataset owner/slug/version")
    from scripts.cloud_prepare import prepare
    derived = work / "derived" / name
    prepared = prepare(config["annotations_root"], derived,
                       config.get("images_root") if task.startswith("v1") else None,
                       {"images": config.get("images_dataset"), "annotations": config["annotations_dataset"]})
    common = {"config": config, "prepared": prepared, "environment": environment_receipt()}
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    common["code_commit"] = code_commit
    from src.train_v1 import code_fingerprint
    common["python_code_sha256"] = code_fingerprint()
    common["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if task.startswith("v1"):
        model_config = yaml.safe_load(Path(config.get("v1_config", "configs/v1_config.yaml")).read_text(encoding="utf-8"))
        merge_overrides(model_config, config.get("v1_overrides", {}))
        model_config["data"].update(master_csv=str(derived / "dataset/dataset_master.csv"),
            nifti_dir=config["images_root"], fold=int(config.get("fold", 1)),
            image_cache_dir=str(Path(config.get("cache_root", "/tmp/mri-cache")) / "views"))
        model_config["data"]["images_manifest_sha256"] = prepared["images_manifest_sha256"]
        model_config["training"]["output_dir"] = str(run_dir)
        if task == "v1-train":
            from src.train_v1 import train
            train(model_config, config.get("resume_from"), config.get("max_runtime_minutes", 60),
                  smoke=config.get("mode", "smoke") == "smoke")
        else:
            from src.infer_v1 import infer
            if not config.get("adapter_path"):
                raise ValueError("v1-infer requires adapter_path")
            run_dir.mkdir(parents=True)
            infer(model_config, config["adapter_path"], run_dir / "predictions.jsonl",
                  split=config.get("split", "test"), limit=2 if config.get("mode") == "smoke" else None)
        write_json(run_dir / "effective_config.json", model_config)
    elif task == "v2-generate":
        from src.generate_report import run_batch
        run_dir.mkdir(parents=True)
        report_config = yaml.safe_load(Path(config.get("v2_config", "configs/v2_report.yaml")).read_text(encoding="utf-8"))
        report_config.update(config.get("report_overrides", {}))
        code = run_batch(derived / "v2/requests.jsonl", run_dir / "reports.jsonl", report_config,
                         max_cases=2 if config.get("mode", "smoke") == "smoke" and report_config["backend"] != "template" else None)
        if code:
            write_json(run_dir / "cloud_receipt.json", {**common, "status": "failed"})
            raise RuntimeError("V2 batch contains failed/invalid cases; inspect summary and receipt")
    else:
        from src.train_v2_report import train_text
        if not config.get("reviewed_targets"):
            raise ValueError("v2-train requires reviewed_targets; raw legacy SFT is not accepted")
        text_config = yaml.safe_load(Path(config.get("v2_train_config", "configs/v2_train.yaml")).read_text(encoding="utf-8"))
        merge_overrides(text_config, config.get("v2_train_overrides", {}))
        text_config["targets"] = config["reviewed_targets"]
        text_config["output_dir"] = str(run_dir)
        text_config["fold"] = config.get("fold", 1)
        train_text(text_config, config.get("resume_from"), config.get("mode") == "smoke",
                   config.get("max_runtime_minutes", 60))
    write_json(run_dir / "cloud_receipt.json", common)
    print(f"Artifacts: {run_dir}. Verify Saved Version outputs before ending your session.")
    return run_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    a = p.parse_args()
    run(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")))
