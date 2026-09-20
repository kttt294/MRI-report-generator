"""Read annotations/images from separate private Kaggle inputs. No source writes."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import file_hash, write_json, strict_loads, content_hash
from scripts.consolidate_dataset import consolidate
from scripts.build_v2_inputs import build


def resolve_annotations_root(annotations_root):
    """Accept a dataset mount or its nested annotations folder, never guess between datasets."""
    source = Path(annotations_root).resolve()
    required = ["grading/grading_all.csv", "localize/disc_localization.csv"]
    def valid(path):
        return all((path / x).is_file() for x in required) and (path / "folds").is_dir()
    if not source.is_dir():
        raise ValueError(f"annotations_root does not exist or is not a directory: {source}. "
                         "Attach the private annotations Dataset and copy its mounted folder path from Kaggle Input.")
    if valid(source):
        return source
    candidates = sorted({p.parent.parent.resolve() for p in source.rglob("grading_all.csv")
                         if p.parent.name == "grading" and valid(p.parent.parent)})
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError("Multiple annotations datasets found; set annotations_root to exactly one: "
                         + ", ".join(map(str, candidates)))
    missing = [x for x in required if not (source / x).is_file()]
    if not (source / "folds").is_dir():
        missing.append("folds/")
    raise ValueError(f"No complete annotations dataset found under {source}. Missing at this root: "
                     + ", ".join(missing) + ". Upload the extracted grading/, localize/, folds/ folders, not only a ZIP or consolidated JSONL.")


def prepare(annotations_root, output, images_root=None, source_versions=None):
    input_root, output = Path(annotations_root).resolve(), Path(output).resolve()
    source = resolve_annotations_root(input_root)
    if output == input_root or output.is_relative_to(input_root) or output.is_relative_to(source):
        raise ValueError("Derived output must be outside annotations input")
    if images_root and output.is_relative_to(Path(images_root).resolve()):
        raise ValueError("Derived output must be outside images input")
    hashes = {str(p.relative_to(source)).replace("\\", "/"): file_hash(p)
              for folder in ("grading", "localize", "folds", "reports_json", "reports_text")
              for p in sorted((source / folder).rglob("*")) if p.is_file()}
    identity = {"source_files": hashes, "dataset_versions": source_versions or {},
                "etl_sha256": file_hash(Path(__file__).with_name("consolidate_dataset.py")),
                "adapter_sha256": file_hash(Path(__file__).resolve().parents[1] / "src/data/v2_adapter.py")}
    receipt_path = output / "prepare_receipt.json"
    # Rebuild rather than reusing partially generated files.
    if receipt_path.exists():
        previous = strict_loads(receipt_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Derived directory belongs to different code/data; choose a new output")
    consolidate(source, output / "dataset")
    receipt = build(output / "dataset/dataset_patients.jsonl", output / "dataset/dataset_master.csv", output / "v2")
    if receipt["errors"]:
        raise ValueError("V2 adapter failed; inspect private build_receipt.json")
    images = []
    if images_root:
        from src.data.mri_images import volume_index, resolve_volume
        index = volume_index(images_root)
        with (output / "dataset/dataset_master.csv").open(encoding="utf-8-sig", newline="") as f:
            volumes = sorted({r["volume"] for r in csv.DictReader(f)})
        for volume in volumes:
            path = resolve_volume(images_root, volume, index)
            images.append({"volume": volume, "bytes": path.stat().st_size, "sha256": file_hash(path)})
    result = {"identity": identity, "annotations_root": str(source), "images": images, "images_manifest_sha256": content_hash(images),
              "master_sha256": file_hash(output / "dataset/dataset_master.csv"), "requests": receipt["requests"]}
    write_json(receipt_path, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations-root", required=True)
    p.add_argument("--images-root")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = prepare(a.annotations_root, a.output, a.images_root)
    print({"requests": result["requests"], "images": len(result["images"])})
