import json
from pathlib import Path

from src.data.validation import Banking77Validator


def main() -> None:

    report_dir = Path("reports/validation")
    report_dir.mkdir(parents=True, exist_ok=True)

    validator = Banking77Validator(
        config_path="configs/data_config.yaml",
        raw_data_dir="data/raw",
    )

    is_valid = validator.validate()

    report = {
       "status": "passed" if is_valid else "failed",
        "valid": is_valid,
        "errors": validator.errors,
        "warnings": validator.warnings,
    }

    report_path = report_dir / "validation_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


    validator.report()

    if not is_valid:
        raise RuntimeError(
            "Banking77 data validation failed. "
            f"see {report_path}for details."
        )




if __name__ == "__main__":
    main()