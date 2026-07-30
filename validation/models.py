from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    severity: str = "error"
    details: str = ""
    max_error: float | None = None
    rows_checked: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "WARN" if self.severity == "warning" else "FAIL"


@dataclass
class ValidationResult:
    name: str
    checks: list[ValidationCheck] = field(default_factory=list)

    def add(
        self,
        name: str,
        passed: bool,
        *,
        severity: str = "error",
        details: str = "",
        max_error: float | None = None,
        rows_checked: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.checks.append(
            ValidationCheck(
                name=name,
                passed=bool(passed),
                severity=severity,
                details=details,
                max_error=max_error,
                rows_checked=rows_checked,
                metadata=metadata or {},
            )
        )

    @property
    def passed(self) -> bool:
        return all(
            check.passed or check.severity == "warning"
            for check in self.checks
        )

    @property
    def failure_count(self) -> int:
        return sum(
            1
            for check in self.checks
            if not check.passed and check.severity != "warning"
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1
            for check in self.checks
            if not check.passed and check.severity == "warning"
        )

    @property
    def max_error(self) -> float:
        errors = [
            check.max_error
            for check in self.checks
            if check.max_error is not None
        ]
        return max(errors, default=0.0)
