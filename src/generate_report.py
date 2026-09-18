"""Usage: python -m src.generate_report --input requests.jsonl --output results.jsonl"""
import argparse
import collections
import time
from pathlib import Path
import yaml

from src.contracts.report_input import ReportRequest
from src.report.pipeline import generate_report
from src.io_utils import strict_loads, write_json, write_jsonl, environment_receipt, file_hash


def run_batch(input_path, output_path, config, max_cases=None):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Output must not overwrite input")
    backend = None
    if config["backend"] == "hf_local":
        if not config.get("model_path"):
            raise ValueError("hf_local requires model_path")
        from src.report.backends.hf_local import HFLocalBackend
        backend = HFLocalBackend(**{k: config[k] for k in ("model_path", "revision", "max_new_tokens",
            "max_input_tokens", "max_time_seconds", "use_4bit", "local_files_only", "adapter_path") if k in config})
    elif config["backend"] != "template":
        raise ValueError("backend must be template or hf_local")
    path = Path(input_path)
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines() if path.suffix == ".jsonl" else [text]
    results, invalid, seen = [], [], set()
    started = time.monotonic()
    for number, line in enumerate(lines, 1):
        if max_cases is not None and len(results) + len(invalid) >= max_cases:
            break
        if not line.strip():
            continue
        try:
            request = ReportRequest.model_validate(strict_loads(line))
            if request.case_id in seen:
                raise ValueError("Duplicate case_id in batch")
            seen.add(request.case_id)
            result = generate_report(request, backend=backend, max_repairs=config.get("max_repairs", 1),
                fallback=config.get("fallback", True), require_verified=config.get("require_verified", False))
            results.append(result.model_dump())
        except (ValueError, TypeError) as exc:
            invalid.append({"line": number, "error": str(exc)})
    write_jsonl(output_path, results)
    summary = {"input_sha256": file_hash(path), "valid_requests": len(results), "invalid_requests": invalid,
        "statuses": dict(collections.Counter(x["status"] for x in results)),
        "raw_attempt_failures": sum(bool(a["errors"]) for r in results for a in r["attempts"]),
        "attempts": sum(len(r["attempts"]) for r in results), "config": config,
        "environment": environment_receipt(), "elapsed_seconds": time.monotonic() - started,
        "requested_limit": max_cases}
    write_json(str(output_path) + ".summary.json", summary)
    print({k: summary[k] for k in ("valid_requests", "statuses", "raw_attempt_failures")})
    return 1 if not results or invalid or any(x["status"] == "failed" for x in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/v2_report.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    return run_batch(args.input, args.output, config)


if __name__ == "__main__":
    raise SystemExit(main())
