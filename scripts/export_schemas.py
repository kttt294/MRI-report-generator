"""Export schemas from the runtime contracts. Python validators add cross-field checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contracts.report_input import ReportRequest
from src.contracts.report_output import ReportDraft, ReportResult
from src.io_utils import write_json


def export(root="schemas"):
    for name, model in (("report_input", ReportRequest), ("report_draft", ReportDraft), ("report_output", ReportResult)):
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        write_json(Path(root) / f"{name}.schema.json", schema)


if __name__ == "__main__":
    export()
