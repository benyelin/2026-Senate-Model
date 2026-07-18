#!/usr/bin/env python3

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

WAREHOUSE_ROOT = (
    PROJECT_ROOT
    / "historical"
    / "senate"
    / "warehouse"
)

PROCESSED_ROOT = WAREHOUSE_ROOT / "processed"
VALIDATION_ROOT = WAREHOUSE_ROOT / "validation"

RESULTS_PATH = (
    PROCESSED_ROOT
    / "senate_historical_election_results_2012_2024.csv"
)

CLASSES_PATH = (
    PROCESSED_ROOT
    / "senate_historical_classes_2012_2024.csv"
)

SERVICE_TERMS_PATH = (
    PROCESSED_ROOT
    / "senate_historical_service_terms.csv"
)

INCUMBENCY_OVERRIDES_PATH = (
    WAREHOUSE_ROOT
    / "raw"
    / "incumbency"
    / "senate_incumbency_overrides_2012_2024.csv"
)

OUTPUT_PATH = (
    PROCESSED_ROOT
    / "senate_historical_election_results_enriched_2012_2024.csv"
)

AUDIT_PATH = (
    VALIDATION_ROOT
    / "senate_historical_incumbency_audit_2012_2024.csv"
)

VALIDATION_PATH = (
    VALIDATION_ROOT
    / "senate_enriched_election_warehouse_validation.csv"
)


NAME_SUFFIXES = {
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "v",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    tokens = [
        token
        for token in text.split()
        if token not in NAME_SUFFIXES
    ]

    return " ".join(tokens)


def surname(value: object) -> str:
    normalized = normalize_name(value)

    if not normalized:
        return ""

    return normalized.split()[-1]


def name_match_score(
    left: object,
    right: object,
) -> float:
    left_name = normalize_name(left)
    right_name = normalize_name(right)

    if not left_name or not right_name:
        return 0.0

    if left_name == right_name:
        return 1.0

    left_tokens = set(left_name.split())
    right_tokens = set(right_name.split())

    token_union = left_tokens | right_tokens

    token_overlap = (
        len(left_tokens & right_tokens)
        / len(token_union)
        if token_union
        else 0.0
    )

    sequence_score = SequenceMatcher(
        None,
        left_name,
        right_name,
    ).ratio()

    surname_bonus = (
        0.15
        if surname(left_name) == surname(right_name)
        else 0.0
    )

    return min(
        1.0,
        (
            0.55 * sequence_score
            + 0.45 * token_overlap
            + surname_bonus
        ),
    )


def classify_candidate_match(
    incumbent_name: object,
    dem_candidate: object,
    gop_candidate: object,
) -> dict[str, object]:
    dem_score = name_match_score(
        incumbent_name,
        dem_candidate,
    )

    gop_score = name_match_score(
        incumbent_name,
        gop_candidate,
    )

    best_party = pd.NA
    best_score = max(
        dem_score,
        gop_score,
    )

    if best_score >= 0.72:
        if dem_score > gop_score:
            best_party = "D"
        elif gop_score > dem_score:
            best_party = "R"

    if best_score >= 0.90:
        match_method = "strong_name_match"
    elif best_score >= 0.72:
        match_method = "probable_name_match"
    elif best_score > 0:
        match_method = "no_confident_match"
    else:
        match_method = "missing_candidate_name"

    return {
        "lineage_dem_candidate_match_score": round(
            dem_score,
            4,
        ),
        "lineage_gop_candidate_match_score": round(
            gop_score,
            4,
        ),
        "lineage_incumbent_candidate_party": best_party,
        "lineage_incumbent_running": (
            pd.notna(best_party)
        ),
        "lineage_candidate_match_method": match_method,
    }


def select_pre_election_officeholder(
    race: pd.Series,
    service_terms: pd.DataFrame,
) -> dict[str, object]:
    reference_date = (
        race["election_date"]
        - pd.Timedelta(days=1)
    )

    matches = service_terms.loc[
        (
            service_terms["seat_id"]
            == race["seat_id"]
        )
        & (
            service_terms["term_start"]
            <= reference_date
        )
        & (
            service_terms["term_end"]
            >= reference_date
        )
    ].copy()

    result = {
        "incumbency_reference_date": reference_date,
        "lineage_officeholder_count": len(matches),
        "lineage_incumbent_name": pd.NA,
        "lineage_incumbent_party": pd.NA,
        "lineage_incumbent_bioguide_id": pd.NA,
        "lineage_incumbent_term_start": pd.NaT,
        "lineage_incumbent_term_end": pd.NaT,
        "lineage_incumbent_term_how": pd.NA,
        "lineage_appointed_incumbent": False,
        "lineage_officeholder_status": (
            "single_match"
            if len(matches) == 1
            else (
                "no_match"
                if len(matches) == 0
                else "multiple_matches"
            )
        ),
    }

    if len(matches) != 1:
        return result

    officeholder = matches.iloc[0]

    term_how = officeholder["term_how"]

    result.update(
        {
            "lineage_incumbent_name": (
                officeholder["display_name"]
            ),
            "lineage_incumbent_party": (
                officeholder["aligned_party"]
            ),
            "lineage_incumbent_bioguide_id": (
                officeholder["bioguide_id"]
            ),
            "lineage_incumbent_term_start": (
                officeholder["term_start"]
            ),
            "lineage_incumbent_term_end": (
                officeholder["term_end"]
            ),
            "lineage_incumbent_term_how": term_how,
            "lineage_appointed_incumbent": (
                term_how == "appointment"
            ),
        }
    )

    return result


def normalize_boolean_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    ).astype("boolean")


def main() -> None:
    for path in [
        RESULTS_PATH,
        CLASSES_PATH,
        SERVICE_TERMS_PATH,
    ]:
        require_file(path)

    results = pd.read_csv(
        RESULTS_PATH,
        parse_dates=["election_date"],
    )

    classes = pd.read_csv(
        CLASSES_PATH,
    )

    service_terms = pd.read_csv(
        SERVICE_TERMS_PATH,
        parse_dates=[
            "term_start",
            "term_end",
        ],
    )

    expected_result_columns = {
        "race_id",
        "cycle",
        "election_date",
        "state",
        "dem_candidate",
        "gop_candidate",
        "incumbent_name",
        "incumbent_party",
        "incumbent_running",
        "open_seat",
    }

    missing_result_columns = (
        expected_result_columns
        - set(results.columns)
    )

    if missing_result_columns:
        raise ValueError(
            "Election results are missing required "
            f"columns: {sorted(missing_result_columns)}"
        )

    expected_class_columns = {
        "race_id",
        "cycle",
        "state",
        "senate_class",
        "seat_id",
        "special_election",
        "election_type",
    }

    missing_class_columns = (
        expected_class_columns
        - set(classes.columns)
    )

    if missing_class_columns:
        raise ValueError(
            "Class file is missing required "
            f"columns: {sorted(missing_class_columns)}"
        )

    if results["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race_id values in results file."
        )

    if classes["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race_id values in class file."
        )

    result_ids = set(
        results["race_id"]
    )

    class_ids = set(
        classes["race_id"]
    )

    missing_from_classes = sorted(
        result_ids - class_ids
    )

    extra_in_classes = sorted(
        class_ids - result_ids
    )

    if missing_from_classes or extra_in_classes:
        raise ValueError(
            "Race-ID mismatch between results and "
            "class files.\n"
            f"Missing from classes: {missing_from_classes}\n"
            f"Extra in classes: {extra_in_classes}"
        )

    class_fields = classes[
        [
            "race_id",
            "senate_class",
            "seat_id",
            "special_election",
            "election_type",
            "class_assignment_method",
            "class_assignment_notes",
        ]
    ].rename(
        columns={
            "senate_class": (
                "assigned_senate_class"
            ),
            "special_election": (
                "assigned_special_election"
            ),
            "election_type": (
                "assigned_election_type"
            ),
        }
    )

    enriched = results.merge(
        class_fields,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    enriched["senate_class"] = (
        enriched[
            "assigned_senate_class"
        ]
        .astype("Int64")
    )

    enriched["special_election"] = (
        enriched[
            "assigned_special_election"
        ]
        .astype("boolean")
    )

    enriched["election_type"] = (
        enriched[
            "assigned_election_type"
        ]
    )

    enriched = enriched.drop(
        columns=[
            "assigned_senate_class",
            "assigned_special_election",
            "assigned_election_type",
        ]
    )

    officeholder_rows = []

    for _, race in enriched.iterrows():
        officeholder_rows.append(
            select_pre_election_officeholder(
                race,
                service_terms,
            )
        )

    officeholders = pd.DataFrame(
        officeholder_rows
    )

    enriched = pd.concat(
        [
            enriched.reset_index(drop=True),
            officeholders.reset_index(drop=True),
        ],
        axis=1,
    )

    candidate_match_rows = []

    for _, race in enriched.iterrows():
        candidate_match_rows.append(
            classify_candidate_match(
                race[
                    "lineage_incumbent_name"
                ],
                race["dem_candidate"],
                race["gop_candidate"],
            )
        )

    candidate_matches = pd.DataFrame(
        candidate_match_rows
    )

    enriched = pd.concat(
        [
            enriched.reset_index(drop=True),
            candidate_matches.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    enriched["lineage_open_seat"] = (
        ~enriched[
            "lineage_incumbent_running"
        ]
    )

    enriched[
        "incumbency_override_applied"
    ] = False

    enriched[
        "incumbency_override_status"
    ] = pd.NA

    enriched[
        "incumbency_override_notes"
    ] = pd.NA

    if INCUMBENCY_OVERRIDES_PATH.exists():
        overrides = pd.read_csv(
            INCUMBENCY_OVERRIDES_PATH
        )

        if overrides["race_id"].duplicated().any():
            raise ValueError(
                "Duplicate race_id values in "
                "incumbency override file."
            )

        unknown_override_ids = sorted(
            set(overrides["race_id"])
            - set(enriched["race_id"])
        )

        if unknown_override_ids:
            raise ValueError(
                "Unknown race IDs in incumbency "
                f"overrides: {unknown_override_ids}"
            )

        overrides[
            "incumbent_running"
        ] = normalize_boolean_series(
            overrides["incumbent_running"]
        )

        override_lookup = overrides.set_index(
            "race_id"
        )

        for race_id, override in (
            override_lookup.iterrows()
        ):
            mask = (
                enriched["race_id"]
                == race_id
            )

            incumbent_running = bool(
                override["incumbent_running"]
            )

            enriched.loc[
                mask,
                "lineage_incumbent_running",
            ] = incumbent_running

            enriched.loc[
                mask,
                "lineage_open_seat",
            ] = not incumbent_running

            enriched.loc[
                mask,
                "lineage_incumbent_candidate_party",
            ] = override[
                "incumbent_candidate_party"
            ]

            enriched.loc[
                mask,
                "lineage_candidate_match_method",
            ] = (
                "manual_override"
                if override["match_status"]
                == "manual_override"
                else "confirmed_name_match"
            )

            enriched.loc[
                mask,
                "incumbency_override_applied",
            ] = True

            enriched.loc[
                mask,
                "incumbency_override_status",
            ] = override["match_status"]

            enriched.loc[
                mask,
                "incumbency_override_notes",
            ] = override["notes"]

    existing_running = normalize_boolean_series(
        enriched["incumbent_running"]
    )

    existing_open = normalize_boolean_series(
        enriched["open_seat"]
    )

    existing_name_available = (
        enriched["incumbent_name"]
        .astype("string")
        .str.strip()
        .notna()
        & enriched["incumbent_name"]
        .astype("string")
        .str.strip()
        .ne("")
    )

    existing_party_available = (
        enriched["incumbent_party"]
        .astype("string")
        .str.strip()
        .notna()
        & enriched["incumbent_party"]
        .astype("string")
        .str.strip()
        .ne("")
    )

    enriched[
        "existing_incumbency_name_available"
    ] = existing_name_available

    enriched[
        "existing_incumbency_party_available"
    ] = existing_party_available

    enriched[
        "existing_incumbency_running_available"
    ] = existing_running.notna()

    enriched[
        "existing_open_seat_available"
    ] = existing_open.notna()

    enriched[
        "audit_existing_name_match_score"
    ] = enriched.apply(
        lambda row: (
            round(
                name_match_score(
                    row["incumbent_name"],
                    row[
                        "lineage_incumbent_name"
                    ],
                ),
                4,
            )
            if pd.notna(
                row["incumbent_name"]
            )
            and str(
                row["incumbent_name"]
            ).strip()
            else pd.NA
        ),
        axis=1,
    )

    enriched[
        "audit_incumbent_name_agrees"
    ] = pd.Series(
        pd.NA,
        index=enriched.index,
        dtype="boolean",
    )

    enriched.loc[
        existing_name_available,
        "audit_incumbent_name_agrees",
    ] = (
        enriched.loc[
            existing_name_available,
            "audit_existing_name_match_score",
        ]
        >= 0.72
    )

    enriched[
        "audit_incumbent_party_agrees"
    ] = pd.Series(
        pd.NA,
        index=enriched.index,
        dtype="boolean",
    )

    enriched.loc[
        existing_party_available,
        "audit_incumbent_party_agrees",
    ] = (
        enriched.loc[
            existing_party_available,
            "incumbent_party",
        ]
        .astype("string")
        .str.strip()
        .str.upper()
        == enriched.loc[
            existing_party_available,
            "lineage_incumbent_party",
        ]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    enriched[
        "audit_incumbent_running_agrees"
    ] = pd.Series(
        pd.NA,
        index=enriched.index,
        dtype="boolean",
    )

    running_available = existing_running.notna()

    enriched.loc[
        running_available,
        "audit_incumbent_running_agrees",
    ] = (
        existing_running.loc[
            running_available
        ]
        == enriched.loc[
            running_available,
            "lineage_incumbent_running",
        ].astype("boolean")
    )

    enriched[
        "audit_open_seat_agrees"
    ] = pd.Series(
        pd.NA,
        index=enriched.index,
        dtype="boolean",
    )

    open_available = existing_open.notna()

    enriched.loc[
        open_available,
        "audit_open_seat_agrees",
    ] = (
        existing_open.loc[
            open_available
        ]
        == enriched.loc[
            open_available,
            "lineage_open_seat",
        ].astype("boolean")
    )

    existing_disagreement = (
        (
            existing_name_available
            & ~enriched[
                "audit_incumbent_name_agrees"
            ].fillna(True)
        )
        | (
            existing_party_available
            & ~enriched[
                "audit_incumbent_party_agrees"
            ].fillna(True)
        )
        | (
            running_available
            & ~enriched[
                "audit_incumbent_running_agrees"
            ].fillna(True)
        )
        | (
            open_available
            & ~enriched[
                "audit_open_seat_agrees"
            ].fillna(True)
        )
    )

    candidate_match_review = enriched[
        "lineage_candidate_match_method"
    ].isin(
        [
            "probable_name_match",
            "missing_candidate_name",
        ]
    )

    enriched["incumbency_audit_flag"] = (
        (
            enriched[
                "lineage_officeholder_status"
            ]
            != "single_match"
        )
        | existing_disagreement
        | candidate_match_review
    )

    audit_columns = [
        "race_id",
        "cycle",
        "election_date",
        "state",
        "senate_class",
        "seat_id",
        "special_election",
        "dem_candidate",
        "gop_candidate",
        "incumbent_name",
        "incumbent_party",
        "incumbent_running",
        "open_seat",
        "lineage_incumbent_name",
        "lineage_incumbent_party",
        "lineage_incumbent_term_how",
        "lineage_appointed_incumbent",
        "lineage_incumbent_running",
        "lineage_open_seat",
        "lineage_incumbent_candidate_party",
        "lineage_dem_candidate_match_score",
        "lineage_gop_candidate_match_score",
        "lineage_candidate_match_method",
        "lineage_officeholder_count",
        "lineage_officeholder_status",
        "audit_existing_name_match_score",
        "audit_incumbent_name_agrees",
        "audit_incumbent_party_agrees",
        "audit_incumbent_running_agrees",
        "audit_open_seat_agrees",
        "incumbency_audit_flag",
    ]

    audit = enriched[
        audit_columns
    ].copy()

    validation = pd.DataFrame(
        [
            {
                "race_rows": len(
                    enriched
                ),
                "unique_race_ids": (
                    enriched[
                        "race_id"
                    ].nunique()
                ),
                "missing_senate_class": int(
                    enriched[
                        "senate_class"
                    ].isna().sum()
                ),
                "missing_seat_id": int(
                    enriched[
                        "seat_id"
                    ].isna().sum()
                ),
                "single_officeholder_matches": int(
                    (
                        enriched[
                            "lineage_officeholder_status"
                        ]
                        == "single_match"
                    ).sum()
                ),
                "no_officeholder_matches": int(
                    (
                        enriched[
                            "lineage_officeholder_status"
                        ]
                        == "no_match"
                    ).sum()
                ),
                "multiple_officeholder_matches": int(
                    (
                        enriched[
                            "lineage_officeholder_status"
                        ]
                        == "multiple_matches"
                    ).sum()
                ),
                "lineage_incumbents_running": int(
                    enriched[
                        "lineage_incumbent_running"
                    ].sum()
                ),
                "lineage_open_seats": int(
                    enriched[
                        "lineage_open_seat"
                    ].sum()
                ),
                "appointed_incumbents_running": int(
                    (
                        enriched[
                            "lineage_appointed_incumbent"
                        ]
                        & enriched[
                            "lineage_incumbent_running"
                        ]
                    ).sum()
                ),
                "existing_name_values": int(
                    existing_name_available.sum()
                ),
                "existing_party_values": int(
                    existing_party_available.sum()
                ),
                "existing_running_values": int(
                    running_available.sum()
                ),
                "existing_open_seat_values": int(
                    open_available.sum()
                ),
                "existing_name_disagreements": int(
                    (
                        existing_name_available
                        & ~enriched[
                            "audit_incumbent_name_agrees"
                        ].fillna(True)
                    ).sum()
                ),
                "existing_party_disagreements": int(
                    (
                        existing_party_available
                        & ~enriched[
                            "audit_incumbent_party_agrees"
                        ].fillna(True)
                    ).sum()
                ),
                "existing_running_disagreements": int(
                    (
                        running_available
                        & ~enriched[
                            "audit_incumbent_running_agrees"
                        ].fillna(True)
                    ).sum()
                ),
                "existing_open_seat_disagreements": int(
                    (
                        open_available
                        & ~enriched[
                            "audit_open_seat_agrees"
                        ].fillna(True)
                    ).sum()
                ),
                "audit_flagged_races": int(
                    enriched[
                        "incumbency_audit_flag"
                    ].sum()
                ),
            }
        ]
    )

    fatal_failures = {
        "missing_senate_class": int(
            enriched[
                "senate_class"
            ].isna().sum()
        ),
        "missing_seat_id": int(
            enriched[
                "seat_id"
            ].isna().sum()
        ),
        "multiple_officeholder_matches": int(
            (
                enriched[
                    "lineage_officeholder_status"
                ]
                == "multiple_matches"
            ).sum()
        ),
    }

    if any(fatal_failures.values()):
        print(
            validation.to_string(
                index=False
            )
        )

        raise ValueError(
            "Enriched warehouse validation "
            f"failed: {fatal_failures}"
        )

    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
        OUTPUT_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    print(
        "Enriched Senate election rows:",
        len(enriched),
    )

    print()
    print("Validation:")
    print(
        validation.to_string(
            index=False
        )
    )

    print()
    print("Flagged incumbency audit rows:")
    print(
        audit.loc[
            audit[
                "incumbency_audit_flag"
            ],
            [
                "race_id",
                "cycle",
                "state",
                "dem_candidate",
                "gop_candidate",
                "incumbent_name",
                "incumbent_party",
                "incumbent_running",
                "open_seat",
                "lineage_incumbent_name",
                "lineage_incumbent_party",
                "lineage_incumbent_term_how",
                "lineage_incumbent_running",
                "lineage_open_seat",
                "lineage_candidate_match_method",
                "audit_existing_name_match_score",
                "audit_incumbent_name_agrees",
                "audit_incumbent_party_agrees",
                "audit_incumbent_running_agrees",
                "audit_open_seat_agrees",
            ],
        ]
        .sort_values(
            [
                "cycle",
                "state",
                "race_id",
            ]
        )
        .to_string(index=False)
    )

    print()
    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", AUDIT_PATH)
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
