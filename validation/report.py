from __future__ import annotations

from pathlib import Path

from .models import ValidationResult


def render_report(
    results: list[ValidationResult],
    *,
    title: str = "SENATE MODEL VALIDATION REPORT",
) -> str:
    width = max(72, len(title) + 8)
    lines = [
        "=" * width,
        title.center(width),
        "=" * width,
        "",
    ]

    for result in results:
        lines.append(result.name)
        lines.append("-" * len(result.name))

        if not result.checks:
            lines.append("WARN  No checks were executed.")
            lines.append("")
            continue

        for check in result.checks:
            marker = {
                "PASS": "PASS",
                "WARN": "WARN",
                "FAIL": "FAIL",
            }[check.status]

            line = f"{marker:4s}  {check.name}"

            annotations = []

            if check.rows_checked is not None:
                annotations.append(f"rows={check.rows_checked:,}")

            if check.max_error is not None:
                annotations.append(f"max_error={check.max_error:.12g}")

            if annotations:
                line += "  [" + ", ".join(annotations) + "]"

            lines.append(line)

            if check.details:
                for detail_line in str(check.details).splitlines():
                    lines.append(f"      {detail_line}")

        lines.append("")

    failures = sum(result.failure_count for result in results)
    warnings = sum(result.warning_count for result in results)
    max_error = max((result.max_error for result in results), default=0.0)
    overall_pass = failures == 0

    lines.extend(
        [
            "=" * width,
            "SUMMARY",
            "=" * width,
            f"Validation groups: {len(results):,}",
            f"Hard failures:     {failures:,}",
            f"Warnings:          {warnings:,}",
            f"Maximum error:     {max_error:.12g}",
            "",
            f"OVERALL STATUS: {'PASS' if overall_pass else 'FAIL'}",
            "",
        ]
    )

    return "\n".join(lines)


def write_report(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
