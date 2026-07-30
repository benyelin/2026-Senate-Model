from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from validation.report import render_report, write_report
from validation.validate_bayesian_pipeline import (
    validate_bayesian_pipeline,
)
from validation.validate_pipeline_consistency import (
    validate_pipeline_consistency,
)
from validation.validate_poll_weights import (
    validate_poll_weights,
)


DEFAULT_OUTPUT_DIR = Path("validation/outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Senate production validation suite."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit nonzero for hard validation failures. "
            "Warnings remain nonfatal."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Maximum permitted floating-point reconstruction error.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for validation reports and diagnostics.",
    )
    return parser.parse_args()


def write_check_table(results, output_path: Path) -> None:
    rows = []

    for validation in results:
        for check in validation.checks:
            rows.append(
                {
                    "validation_group": validation.name,
                    "check_name": check.name,
                    "status": check.status,
                    "severity": check.severity,
                    "passed": check.passed,
                    "rows_checked": check.rows_checked,
                    "max_error": check.max_error,
                    "details": check.details,
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        validate_poll_weights(args.tolerance),
        validate_bayesian_pipeline(args.tolerance),
        validate_pipeline_consistency(args.tolerance),
    ]

    report = render_report(results)

    report_path = args.output_dir / "validation_summary.txt"
    checks_path = args.output_dir / "validation_checks.csv"

    write_report(report, report_path)
    write_check_table(results, checks_path)

    print(report)
    print(f"Wrote report: {report_path}")
    print(f"Wrote checks: {checks_path}")

    hard_failures = sum(result.failure_count for result in results)

    if args.strict and hard_failures:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
