"""Inspect data without loading a model. --metadata-only does not approve images."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.train_v1 import make_dataset
from src.io_utils import write_json


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/v1_config.yaml")
    p.add_argument("--output", default="output/v1_preflight.json")
    p.add_argument("--metadata-only", action="store_true")
    a = p.parse_args()
    config = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    datasets = [make_dataset(config, split, not a.metadata_only) for split in ("train", "val", "test")]
    if not a.metadata_only:
        for dataset in datasets:
            for i in range(len(dataset)):
                dataset[i]
    write_json(a.output, {"images_checked": not a.metadata_only, "splits": [d.manifest() for d in datasets]})
    print({d.split: len(d) for d in datasets})
