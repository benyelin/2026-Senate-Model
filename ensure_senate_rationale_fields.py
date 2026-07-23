from pathlib import Path
from datetime import date
import pandas as pd

path = Path("inputs/race_inputs.csv")

if not path.exists():
    raise FileNotFoundError("inputs/race_inputs.csv not found.")

df = pd.read_csv(path)

rationale_cols = {
    "incumbency_rationale": "",
    "candidate_quality_rationale": "",
    "overperformance_rationale": "",
    "liability_rationale": "",
    "special_adjustment_rationale": "",
    "last_human_review_date": "",
    "human_review_status": "",
}

for col, default in rationale_cols.items():
    if col not in df.columns:
        df[col] = default

# Fill Maine rationale if blank.
state_col = "state"
if state_col in df.columns:
    me = df[state_col].astype(str).str.upper().eq("ME")

    today = date.today().isoformat()

    def fill_if_blank(col, value):
        if col in df.columns:
            df[col] = df[col].astype("object")
            blank = (
                df[col].isna()
                | df[col].astype(str).str.strip().isin(["", "nan", "None"])
            )
            df.loc[me & blank, col] = value

    fill_if_blank(
        "incumbency_rationale",
        "Intentional exceptional-case adjustment. Susan Collins has a long record of crossover overperformance in Maine; generic incumbency is retained but should be reviewed alongside overperformance and candidate-quality adjustments.",
    )

    fill_if_blank(
        "candidate_quality_rationale",
        "Susan Collins receives a candidate-strength adjustment based on her historical Senate performance and prior statewide victories. Troy Jackson has prior elected-office experience, and no candidate liability or scandal adjustment is currently applied. Review potential overlap between Collins' candidate-strength and incumbency adjustments as additional Jackson-specific polling becomes available.",
    )

    fill_if_blank(
        "overperformance_rationale",
        "Collins has historically overperformed Maine's partisan baseline by a meaningful margin. Kept as a separate adjustment because this race is unusually candidate-specific, but double-counting risk should be monitored.",
    )

    fill_if_blank(
        "liability_rationale",
        "No current candidate-specific liability adjustment is applied to Troy Jackson or Susan Collins.",
    )

    fill_if_blank(
        "human_review_status",
        "Reviewed - intentional exception",
    )

    fill_if_blank(
        "last_human_review_date",
        today,
    )

df.to_csv(path, index=False)

print(f"Updated {path}")
print()
show_cols = [
    "state",
    "dem_candidate",
    "gop_candidate",
    "incumbency_rationale",
    "candidate_quality_rationale",
    "overperformance_rationale",
    "liability_rationale",
    "last_human_review_date",
    "human_review_status",
]
show_cols = [c for c in show_cols if c in df.columns]
print(df[df["state"].astype(str).str.upper().eq("ME")][show_cols].to_string(index=False))
