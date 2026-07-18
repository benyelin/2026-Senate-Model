#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

WAREHOUSE_ROOT = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
)

RAW_ROOT = (
    WAREHOUSE_ROOT
    / "raw"
    / "incumbency"
)

HISTORICAL_PATH = (
    RAW_ROOT
    / "legislators-historical.json"
)

CURRENT_PATH = (
    RAW_ROOT
    / "legislators-current.json"
)

OUTPUT_PATH = (
    WAREHOUSE_ROOT
    / "processed"
    / "senate_historical_service_terms.csv"
)

VALIDATION_PATH = (
    WAREHOUSE_ROOT
    / "validation"
    / "senate_historical_service_terms_validation.csv"
)


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required source file not found:\n{path}"
        )

    with path.open() as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list in {path.name}"
        )

    return data


def clean_text(value: object) -> object:
    if value is None:
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    return text


def normalize_party(
    party: object,
    caucus: object,
) -> str:
    party_text = (
        str(party).strip().lower()
        if party is not None
        else ""
    )

    caucus_text = (
        str(caucus).strip().lower()
        if caucus is not None
        else ""
    )

    if party_text.startswith("democrat"):
        return "D"

    if party_text.startswith("republican"):
        return "R"

    if party_text == "independent":
        if caucus_text.startswith("democrat"):
            return "D"

        if caucus_text.startswith("republican"):
            return "R"

        return "I"

    return "Other"


def build_display_name(
    name: dict,
) -> str:
    official = clean_text(
        name.get("official_full")
    )

    if pd.notna(official):
        return str(official)

    parts = [
        clean_text(name.get("first")),
        clean_text(name.get("middle")),
        clean_text(name.get("nickname")),
        clean_text(name.get("last")),
        clean_text(name.get("suffix")),
    ]

    return " ".join(
        str(part)
        for part in parts
        if pd.notna(part)
    )


def flatten_legislators(
    legislators: list[dict],
    source_file: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for legislator in legislators:
        identifiers = legislator.get(
            "id",
            {}
        )

        name = legislator.get(
            "name",
            {}
        )

        display_name = build_display_name(
            name
        )

        for term_number, term in enumerate(
            legislator.get("terms", []),
            start=1,
        ):
            if term.get("type") != "sen":
                continue

            party = clean_text(
                term.get("party")
            )

            caucus = clean_text(
                term.get("caucus")
            )

            rows.append(
                {
                    "bioguide_id": clean_text(
                        identifiers.get(
                            "bioguide"
                        )
                    ),
                    "lis_id": clean_text(
                        identifiers.get("lis")
                    ),
                    "govtrack_id": clean_text(
                        identifiers.get(
                            "govtrack"
                        )
                    ),
                    "fec_ids": "|".join(
                        identifiers.get("fec", [])
                    )
                    if identifiers.get("fec")
                    else pd.NA,
                    "display_name": display_name,
                    "first_name": clean_text(
                        name.get("first")
                    ),
                    "middle_name": clean_text(
                        name.get("middle")
                    ),
                    "last_name": clean_text(
                        name.get("last")
                    ),
                    "suffix": clean_text(
                        name.get("suffix")
                    ),
                    "nickname": clean_text(
                        name.get("nickname")
                    ),
                    "term_number": term_number,
                    "term_start": clean_text(
                        term.get("start")
                    ),
                    "term_end": clean_text(
                        term.get("end")
                    ),
                    "state": clean_text(
                        term.get("state")
                    ),
                    "senate_class": (
                        term.get("class")
                    ),
                    "party_raw": party,
                    "caucus_raw": caucus,
                    "aligned_party": (
                        normalize_party(
                            party,
                            caucus,
                        )
                    ),
                    "term_how": clean_text(
                        term.get("how")
                    ),
                    "term_end_type": clean_text(
                        term.get("end-type")
                    ),
                    "state_rank": clean_text(
                        term.get("state_rank")
                    ),
                    "source_file": source_file,
                }
            )

    return rows


def main() -> None:
    historical = load_json(
        HISTORICAL_PATH
    )

    current = load_json(
        CURRENT_PATH
    )

    rows = []

    rows.extend(
        flatten_legislators(
            historical,
            HISTORICAL_PATH.name,
        )
    )

    rows.extend(
        flatten_legislators(
            current,
            CURRENT_PATH.name,
        )
    )

    output = pd.DataFrame(rows)

    if output.empty:
        raise ValueError(
            "No Senate service terms were found."
        )

    output["term_start"] = pd.to_datetime(
        output["term_start"],
        errors="coerce",
    )

    output["term_end"] = pd.to_datetime(
        output["term_end"],
        errors="coerce",
    )

    output["senate_class"] = pd.to_numeric(
        output["senate_class"],
        errors="coerce",
    ).astype("Int64")

    output["state"] = (
        output["state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    output["seat_id"] = (
        output["state"]
        + "_CLASS_"
        + output[
            "senate_class"
        ].astype("string")
    )

    output = (
        output.sort_values(
            [
                "state",
                "senate_class",
                "term_start",
                "display_name",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_terms = output.duplicated(
        subset=[
            "bioguide_id",
            "term_start",
            "term_end",
            "state",
            "senate_class",
        ],
        keep=False,
    )

    invalid_dates = output.loc[
        output["term_start"].isna()
        | output["term_end"].isna()
        | (
            output["term_end"]
            < output["term_start"]
        )
    ]

    invalid_classes = output.loc[
        ~output["senate_class"].isin(
            [1, 2, 3]
        )
    ]

    missing_states = output.loc[
        output["state"].isna()
    ]

    validation = pd.DataFrame(
        [
            {
                "service_term_rows": len(
                    output
                ),
                "unique_senators": (
                    output[
                        "bioguide_id"
                    ].nunique()
                ),
                "unique_seats": (
                    output[
                        "seat_id"
                    ].nunique()
                ),
                "terms_starting_2000_or_later": int(
                    (
                        output[
                            "term_start"
                        ].dt.year
                        >= 2000
                    ).sum()
                ),
                "appointment_terms": int(
                    (
                        output[
                            "term_how"
                        ]
                        == "appointment"
                    ).sum()
                ),
                "special_election_terms": int(
                    (
                        output[
                            "term_how"
                        ]
                        == "special-election"
                    ).sum()
                ),
                "duplicate_terms": int(
                    duplicate_terms.sum()
                ),
                "invalid_dates": len(
                    invalid_dates
                ),
                "invalid_classes": len(
                    invalid_classes
                ),
                "missing_states": len(
                    missing_states
                ),
                "missing_bioguide_ids": int(
                    output[
                        "bioguide_id"
                    ].isna().sum()
                ),
            }
        ]
    )

    fatal_failures = {
        "duplicate_terms": int(
            duplicate_terms.sum()
        ),
        "invalid_dates": len(
            invalid_dates
        ),
        "invalid_classes": len(
            invalid_classes
        ),
        "missing_states": len(
            missing_states
        ),
    }

    if any(fatal_failures.values()):
        print(
            validation.to_string(
                index=False
            )
        )

        raise ValueError(
            "Senate service-term validation "
            f"failed: {fatal_failures}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    print(
        "Senate service terms:",
        len(output),
    )

    print(
        "Unique senators:",
        output[
            "bioguide_id"
        ].nunique(),
    )

    print(
        "Unique Senate seats:",
        output[
            "seat_id"
        ].nunique(),
    )

    print()
    print("Validation:")
    print(
        validation.to_string(
            index=False
        )
    )

    print()
    print("Recent appointment terms:")
    print(
        output.loc[
            (
                output["term_how"]
                == "appointment"
            )
            & (
                output[
                    "term_start"
                ].dt.year
                >= 2000
            ),
            [
                "display_name",
                "state",
                "senate_class",
                "term_start",
                "term_end",
                "party_raw",
                "aligned_party",
                "term_how",
            ],
        ]
        .sort_values("term_start")
        .to_string(index=False)
    )

    print()
    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", VALIDATION_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise
