from pathlib import Path
from datetime import date
import argparse

import pandas as pd


INPUTS = Path("inputs")
NATIONAL_ENV_PATH = INPUTS / "national_environment.csv"

GENERIC_BALLOT_COEFFICIENT = 0.90


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update the generic-ballot-based national environment "
            "used by the Senate model and downstream House model."
        )
    )

    parser.add_argument(
        "--as-of-date",
        default=date.today().isoformat(),
        help="Date for this national environment update, YYYY-MM-DD.",
    )

    parser.add_argument(
        "--generic-ballot",
        type=float,
        required=True,
        help="Generic ballot margin, Democratic minus Republican.",
    )

    parser.add_argument(
        "--notes",
        default=(
            "Manual update from latest generic congressional "
            "ballot average."
        ),
        help="Source notes for dashboard and audit trail.",
    )

    args = parser.parse_args()

    INPUTS.mkdir(parents=True, exist_ok=True)

    national_environment_margin_dem = (
        GENERIC_BALLOT_COEFFICIENT
        * args.generic_ballot
    )

    # Preserve the established CSV schema for compatibility with
    # downstream Senate and House readers. Approval and midterm fields
    # are now informationally inactive and do not enter the formula.
    row = {
        "as_of_date": args.as_of_date,
        "generic_ballot_margin_dem": args.generic_ballot,
        "presidential_approval": pd.NA,
        "presidential_disapproval": pd.NA,
        "presidential_net_approval": pd.NA,
        "president_party": pd.NA,
        "midterm_adjustment_dem": 0.0,
        "approval_adjustment_dem": 0.0,
        "national_environment_margin_dem": (
            national_environment_margin_dem
        ),
        "source_notes": args.notes,
    }

    pd.DataFrame([row]).to_csv(
        NATIONAL_ENV_PATH,
        index=False,
    )

    print(f"Updated {NATIONAL_ENV_PATH}")
    print()
    print("National environment calculation:")
    print("  Formula: 0.90 * generic ballot")
    print(
        "  Generic ballot margin Dem:     "
        f"{args.generic_ballot:+.2f} x "
        f"{GENERIC_BALLOT_COEFFICIENT:.2f} = "
        f"{national_environment_margin_dem:+.2f}"
    )
    print(
        "  Presidential approval:         "
        "not used"
    )
    print(
        "  Standalone midterm adjustment: "
        "not used"
    )
    print(
        "  National environment Dem:      "
        f"{national_environment_margin_dem:+.2f}"
    )


if __name__ == "__main__":
    main()
