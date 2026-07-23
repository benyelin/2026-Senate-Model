from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path("inputs/race_inputs.csv")

# Democratic presidential margins.
# Positive = Democratic margin.
# Negative = Republican margin.
#
# Derived from Democratic % minus Republican %.
PRESIDENTIAL_MARGINS = {
    "AK": {
        "pres_2024_margin_dem": 41.41 - 54.54,  # -13.13
        "pres_2020_margin_dem": 42.77 - 52.83,  # -10.06
        "pres_2016_margin_dem": 36.6 - 51.3,    # -14.70
        "state_elasticity": 0.65,
    },
    "FL": {
        "pres_2024_margin_dem": 42.99 - 56.09,  # -13.10
        "pres_2020_margin_dem": 47.86 - 51.22,  # -3.36
        "pres_2016_margin_dem": 47.8 - 49.0,    # -1.20
        "state_elasticity": 0.70,
    },
    "GA": {
        "pres_2024_margin_dem": 48.53 - 50.73,  # -2.20
        "pres_2020_margin_dem": 49.50 - 49.26,  # +0.24
        "pres_2016_margin_dem": 45.9 - 51.0,    # -5.10
        "state_elasticity": 0.90,
    },
    "ME": {
        "pres_2024_margin_dem": 52.44 - 45.50,  # +6.94
        "pres_2020_margin_dem": 53.09 - 44.02,  # +9.07
        "pres_2016_margin_dem": 47.8 - 44.9,    # +2.90
        "state_elasticity": 0.75,
    },
    "NC": {
        "pres_2024_margin_dem": 47.65 - 50.86,  # -3.21
        "pres_2020_margin_dem": 48.59 - 49.93,  # -1.34
        "pres_2016_margin_dem": 46.2 - 49.8,    # -3.60
        "state_elasticity": 0.90,
    },
    "OH": {
        "pres_2024_margin_dem": 43.93 - 55.14,  # -11.21
        "pres_2020_margin_dem": 45.24 - 53.27,  # -8.03
        "pres_2016_margin_dem": 43.6 - 51.7,    # -8.10
        "state_elasticity": 0.75,
    },
    "TX": {
        "pres_2024_margin_dem": 42.46 - 56.14,  # -13.68
        "pres_2020_margin_dem": 46.48 - 52.06,  # -5.58
        "pres_2016_margin_dem": 43.2 - 52.2,    # -9.00
        "state_elasticity": 0.75,
    },
}

# Race-specific adjustments.
# Positive helps Democrats; negative helps Republicans.
#
# These are intentionally modest placeholders.
# We can tune later after the model output passes a smell test.
RACE_ADJUSTMENTS = {
    # Sullivan incumbent; RCV/top-four adds uncertainty elsewhere, not a margin boost here.
    "AK": {
        "incumbency_adjustment_dem": -1.50,
        "candidate_quality_adjustment_dem": 1.00,  # Peltola-style crossover strength placeholder
        "special_adjustment_dem": 0.00,
        "fundamentals_notes_extra": "AK: GOP incumbent advantage partly offset by Peltola/crossover-style Democratic candidate strength placeholder.",
    },

    # Moody is appointed incumbent; GOP state baseline is strong.
    "FL": {
        "incumbency_adjustment_dem": -1.00,
        "candidate_quality_adjustment_dem": 0.00,
        "special_adjustment_dem": 0.00,
        "fundamentals_notes_extra": "FL: appointed GOP incumbent advantage placeholder.",
    },

    # Georgia is now a very competitive presidential state; keep race adjustment neutral for now.
    "GA": {
        "incumbency_adjustment_dem": 0.00,
        "candidate_quality_adjustment_dem": 0.00,
        "special_adjustment_dem": 0.00,
        "fundamentals_notes_extra": "GA: neutral race-specific adjustment pending candidate-field clarity.",
    },

    # Collins historically overperforms presidential baseline; review overlap with candidate-strength framework.
    "ME": {
        "incumbency_adjustment_dem": -2.50,
        "candidate_quality_adjustment_dem": 0.00,
        "special_adjustment_dem": 0.00,
        "fundamentals_notes_extra": "ME: Collins incumbent overperformance placeholder.",
    },

    # Cooper is unusually strong for NC; GOP nominee uncertainty.
    "NC": {
        "incumbency_adjustment_dem": 0.00,
        "candidate_quality_adjustment_dem": 2.00,
        "special_adjustment_dem": 0.00,
        "fundamentals_notes_extra": "NC: Cooper candidate-strength placeholder.",
    },

    # Brown is a strong candidate historically, but OH is now red-leaning.
    "OH": {
        "incumbency_adjustment_dem": 0.00,
        "candidate_quality_adjustment_dem": 2.00,
        "special_adjustment_dem": 0.00,
        "fundamentals_notes_extra": "OH: Brown candidate-strength placeholder against red-state baseline.",
    },

    # Texas is still structurally GOP; Talarico gets a modest candidate-quality placeholder.
    "TX": {
        "incumbency_adjustment_dem": 0.00,
        "candidate_quality_adjustment_dem": 1.00,
        "special_adjustment_dem": 0.00,
        "fundamentals_notes_extra": "TX: modest Democratic candidate-quality placeholder; GOP structural lean retained.",
    },
}


def ensure_columns(df):
    needed = [
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_elasticity",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_notes",
    ]

    for col in needed:
        if col not in df.columns:
            if col == "fundamentals_notes":
                df[col] = ""
            else:
                df[col] = np.nan

    return df


def main():
    if not PATH.exists():
        raise FileNotFoundError(f"Could not find {PATH}")

    df = pd.read_csv(PATH)
    df = ensure_columns(df)

    if "state" not in df.columns:
        raise ValueError("race_inputs.csv must include a state column")

    df["state"] = df["state"].astype(str).str.strip().str.upper()

    for state, vals in PRESIDENTIAL_MARGINS.items():
        mask = df["state"] == state

        if not mask.any():
            print(f"Warning: state {state} not found in race_inputs.csv")
            continue

        for col, val in vals.items():
            df.loc[mask, col] = val

    for state, vals in RACE_ADJUSTMENTS.items():
        mask = df["state"] == state

        if not mask.any():
            continue

        for col in [
            "incumbency_adjustment_dem",
            "candidate_quality_adjustment_dem",
            "special_adjustment_dem",
        ]:
            df.loc[mask, col] = vals.get(col, 0.0)

        existing_notes = df.loc[mask, "fundamentals_notes"].fillna("").astype(str)
        extra = vals.get("fundamentals_notes_extra", "")

        df.loc[mask, "fundamentals_notes"] = existing_notes.apply(
            lambda x: (x + " " + extra).strip() if x else extra
        )

    df.to_csv(PATH, index=False)

    show_cols = [
        "state",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_elasticity",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]

    print("Populated presidential margins and race adjustments for key states:")
    print(df[df["state"].isin(PRESIDENTIAL_MARGINS.keys())][show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
