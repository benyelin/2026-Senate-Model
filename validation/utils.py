from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TOLERANCE = 1e-9


def read_csv(path: str | Path, *, required: bool = True) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()

    frame = pd.read_csv(path, low_memory=False)

    if "state" in frame.columns:
        frame = frame.copy()
        frame["state"] = (
            frame["state"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    return frame


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def maximum_absolute_error(
    actual: pd.Series,
    expected: pd.Series,
) -> tuple[float, int]:
    actual_num = numeric(actual)
    expected_num = numeric(expected)

    comparable = actual_num.notna() & expected_num.notna()

    if not comparable.any():
        return 0.0, 0

    errors = (actual_num[comparable] - expected_num[comparable]).abs()

    return float(errors.max()), int(comparable.sum())


def available_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def duplicate_key_count(
    frame: pd.DataFrame,
    columns: list[str],
) -> int:
    if frame.empty or not set(columns).issubset(frame.columns):
        return 0
    return int(frame.duplicated(columns, keep=False).sum())


def merge_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: str = "state",
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    if key not in left.columns:
        raise KeyError(f"Left table is missing key column: {key}")
    if key not in right.columns:
        raise KeyError(f"Right table is missing key column: {key}")

    return left.merge(
        right,
        on=key,
        how="outer",
        suffixes=suffixes,
        indicator=True,
        validate="one_to_one",
    )


def finite_or_missing(series: pd.Series) -> bool:
    values = numeric(series).dropna()
    return bool(np.isfinite(values).all())
