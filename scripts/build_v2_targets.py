"""Create review packets or assemble approved records. Never auto-approve labels."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import strict_loads, content_hash, write_jsonl
from src.contracts.report_input import ReportRequest
from src.contracts.report_output import ReportDraft
from src.report.planner import make_plan, template_draft
from src.report.validate import validate_draft


def build(requests, manifest, output, reviewed=None):
    manifests = {r["case_id"]: r for r in map(strict_loads, Path(manifest).read_text(encoding="utf-8").splitlines())}
    reviewed_rows = [strict_loads(x) for x in Path(reviewed).read_text(encoding="utf-8").splitlines() if x.strip()] if reviewed else []
    approvals = {x["case_id"]: x for x in reviewed_rows}
    if len(approvals) != len(reviewed_rows): raise ValueError("Duplicate reviewed case")
    result, seen = [], set()
    for line in Path(requests).read_text(encoding="utf-8").splitlines():
        req = ReportRequest.model_validate(strict_loads(line))
        if req.case_id in seen: raise ValueError("Duplicate request case")
        seen.add(req.case_id)
        row = {"case_id": req.case_id, "request": req.model_dump(), "input_sha256": content_hash(req.model_dump()),
               "folds": manifests[req.case_id]["folds"], "target": template_draft(make_plan(req)).model_dump(),
               "review_status": "pending", "reviewed_by": "", "review_notes": "",
               "target_kind": "template_draft_not_clinical_ground_truth"}
        if reviewed:
            if req.case_id not in approvals: continue
            approved = approvals[req.case_id]
            if approved.get("review_status") != "accepted" or not approved.get("reviewed_by"):
                raise ValueError("Each selected review must be explicitly accepted by a reviewer")
            if approved.get("input_sha256") != row["input_sha256"]:
                raise ValueError("Review references a changed input")
            _, errors = validate_draft(ReportDraft.model_validate(approved["target"]).model_dump_json(), make_plan(req))
            if errors:
                raise ValueError("Reviewed target violates the controlled R3 policy: " + "; ".join(errors))
            row.update(target=ReportDraft.model_validate(approved["target"]).model_dump(),
                review_status="accepted", reviewed_by=approved["reviewed_by"],
                review_notes=approved.get("review_notes", ""), target_kind="human_reviewed_scoped")
        result.append(row)
    if set(approvals) - seen: raise ValueError("Review has case not present in requests")
    write_jsonl(output, result)
    return len(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reviewed")
    a = p.parse_args()
    print("Records:", build(a.requests, a.manifest, a.output, a.reviewed))
