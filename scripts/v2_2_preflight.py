"""Inspect V2-2 data and token boundaries before allocating GPU memory."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.v2_2_dataset import load_samples
from src.io_utils import write_json, file_hash


def preflight(patients, master, fold, output, tokenizer_name=None, max_sequence_length=4096):
    by_split, counts = {}, {}
    for split in ("train", "val", "test"):
        by_split[split], counts[split] = load_samples(patients, master, fold, split)
    ids = [set(x["case_id"] for x in by_split[s]) for s in ("train", "val", "test")]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Patient leakage across splits")
    receipt = {"fold": fold, "counts": counts, "master_sha256": file_hash(master),
               "patients_sha256": file_hash(patients), "target_kind": "original_doctor_report",
               "scope_warning": "Original reports may contain facts not represented by the 8 grading fields."}
    if tokenizer_name:
        from transformers import AutoTokenizer
        from src.data.v2_dataset import TextCollator
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        collator = TextCollator(tokenizer, max_sequence_length)
        lengths = {}
        for split in ("train", "val"):
            lengths[split] = []
            for row in by_split[split]:
                batch = collator([row])
                lengths[split].append(batch["input_ids"].shape[-1])
        receipt["tokens"] = {s: {"max": max(v), "min": min(v), "count": len(v)} for s, v in lengths.items()}
    write_json(output, receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("patients", "master", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--tokenizer-name")
    p.add_argument("--max-sequence-length", type=int, default=4096)
    args = p.parse_args()
    print(preflight(**vars(args)))
