import json

import pytest

from scripts.compare_v2_2_qwen3 import (find_frozen_file, prompt_manifest, summarize,
                                        verify_samples, write_comparison)
from src.io_utils import content_hash, file_hash


def sample_pair():
    sample = {"case_id": "test-1", "prompt": "same prompt", "input_sha256": "input",
              "target_sha256": "target", "completion": '{"findings":"F","impression":"I"}'}
    pred = {"case_id": "test-1", "input_sha256": "input", "target_sha256": "target",
            "status": "invalid_json", "parsed": None, "raw_output": "unfinished",
            "hit_token_limit": True, "seconds": 10.0, "new_tokens": 1536}
    config = {"test_cases": 1, "prompt_manifest_sha256": content_hash(prompt_manifest([sample]))}
    return sample, pred, config


def test_cached_comparison_rejects_prompt_target_or_cohort_changes():
    sample, pred, config = sample_pair()
    assert verify_samples([sample], [pred], config)[0]["prediction"] is None
    for key in ("prompt", "target_sha256", "input_sha256"):
        with pytest.raises(ValueError, match="prompts or targets"):
            verify_samples([{**sample, key: "changed"}], [pred], config)
    with pytest.raises(ValueError, match="case count"):
        verify_samples([], [pred], config)
    with pytest.raises(ValueError, match="Duplicate or unmatched"):
        verify_samples([sample], [pred, pred], config)


def test_discovery_uses_content_hash_not_dataset_folder_name(tmp_path):
    good = tmp_path / "nested" / "predictions.jsonl"
    good.parent.mkdir()
    good.write_text("right version")
    (tmp_path / "predictions.jsonl").write_text("wrong version")
    assert find_frozen_file(tmp_path, "predictions.jsonl", file_hash(good)) == good
    with pytest.raises(ValueError, match="Missing frozen"):
        find_frozen_file(tmp_path, "predictions.jsonl", "wrong-hash")


def test_summary_keeps_failures_and_review_includes_raw_output(tmp_path):
    import csv
    sample, pred, config = sample_pair()
    records = verify_samples([sample], [pred], config)
    metrics = summarize(records, [pred])
    assert metrics["format_valid_rate"] == 0
    assert metrics["hit_token_limit"] == 1
    assert metrics["mean_generation_seconds"] == 10
    metrics.update(bleu4={"score": 0}, rouge1_f1=0, rougeL_f1=0, bertscore_f1=0)
    write_comparison(tmp_path, {"baseline": metrics, "candidate": metrics}, records, records)
    with (tmp_path / "human_review.csv").open(encoding="utf-8-sig", newline="") as f:
        review = list(csv.DictReader(f))
    assert review[0]["finetuned_findings"] == ""
    assert review[0]["finetuned_raw_output"] == "unfinished"
    assert review[0]["qwen3_raw_output"] == "unfinished"
    assert review[0]["reviewer"] == ""
