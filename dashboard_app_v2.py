from pathlib import Path
from datetime import datetime
import subprocess
import sys
import tempfile

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from pollster_registry import apply_pollster_registry


SHARED_MODEL_ROOT = Path(
    "/Users/benyelin/Developer/election_model_shared"
)

if str(SHARED_MODEL_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SHARED_MODEL_ROOT),
    )

from candidate_event_dashboard import (
    render_candidate_event_registry_editor,
)

DEM_COLOR = "#1f77b4"
GOP_COLOR = "#d62728"
SENATE_CONTROL_THRESHOLD = 51


INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

st.set_page_config(
    page_title="2026 Senate Forecast Dashboard V2",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY"
}

STATE_NAMES_TO_CODES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY"
}


def infer_state_from_race_label(race):
    import re

    text = str(race).strip().upper()

    # "OH Senate", "ME Senate", etc.
    match = re.match(r"^([A-Z]{2})\b", text)
    if match and match.group(1) in STATE_CODES:
        return match.group(1)

    # "Ohio Senate", "North Carolina Senate", etc.
    for name, code in STATE_NAMES_TO_CODES.items():
        if text.startswith(name + " "):
            return code

    return None

def read_csv_safe(path):
    try:
        if Path(path).exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
    return pd.DataFrame()


def as_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt_margin(x):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def fmt_pct(x):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    return f"{x:.1%}"


def fmt_num(x, digits=2):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def fmt_seat_range(p25, p75):
    p25 = as_float(p25)
    p75 = as_float(p75)
    if pd.isna(p25) or pd.isna(p75):
        return "—"
    return f"{p25:.0f}–{p75:.0f}"


def fmt_margin_range(p25, p75):
    return f"{fmt_margin(p25)} to {fmt_margin(p75)}"


def normalize_state(df):
    if not df.empty and "state" in df.columns:
        df = df.copy()
        df["state"] = df["state"].astype(str).str.strip().str.upper()
    return df


def race_rating_from_prob(p):
    p = as_float(p)
    if pd.isna(p):
        return "Unknown"

    if p >= 0.95:
        return "Safe D"
    if p >= 0.85:
        return "Likely D"
    if p >= 0.65:
        return "Lean D"
    if p >= 0.55:
        return "Tilt D"
    if p > 0.45:
        return "Toss-up"
    if p > 0.35:
        return "Tilt R"
    if p > 0.15:
        return "Lean R"
    if p > 0.05:
        return "Likely R"
    return "Safe R"


def load_data():
    data = {
        "summary": read_csv_safe(OUTPUTS / "forecast_summary.csv"),
        "race_stats": normalize_state(read_csv_safe(OUTPUTS / "race_stats.csv")),
        "seat_distribution": read_csv_safe(OUTPUTS / "seat_distribution.csv"),
        "scenarios": read_csv_safe(OUTPUTS / "scenario_summary.csv"),
        "race_inputs": normalize_state(read_csv_safe(INPUTS / "race_inputs.csv")),
        "polling": normalize_state(read_csv_safe(INPUTS / "polling_averages_generated.csv")),
        "bayes": normalize_state(read_csv_safe(INPUTS / "bayesian_update_generated.csv")),
        "national_env": read_csv_safe(INPUTS / "national_environment.csv"),
        "forecast_history": read_csv_safe(OUTPUTS / "senate_forecast_history.csv"),
    }
    return data


data = load_data()

summary = data["summary"]
race_stats = data["race_stats"]
seat_distribution = data["seat_distribution"]
scenarios = data["scenarios"]
race_inputs = data["race_inputs"]
polling = data["polling"]
bayes = data["bayes"]
national_env = data["national_env"]
forecast_history = data.get("forecast_history", pd.DataFrame())

# -----------------------------
# Header
# -----------------------------
st.title("2026 Senate Forecast Dashboard")
st.caption("Clean view: forecast overview, race ratings, model drivers, scenarios, and diagnostics.")

if summary.empty:
    st.error("No forecast summary found. Run `python3 run_full_pipeline.py` first.")
    st.stop()

summary_row = summary.iloc[-1].to_dict()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_races, tab_drivers, tab_scenarios, tab_manual_polls, tab_diagnostics = st.tabs(
    [
        "Overview",
        "Race Ratings",
        "Model Drivers",
        "Scenarios",
        "Manual Poll Entry",
        "Diagnostics",
    ]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.subheader("Topline Forecast")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Dem Control Odds", fmt_pct(summary_row.get("dem_control_probability")))
    c2.metric("Expected Dem Seats", fmt_num(summary_row.get("expected_dem_seats"), 2))
    c3.metric("Median Dem Seats", fmt_num(summary_row.get("median_dem_seats"), 0))
    c4.metric("Middle 50% Seats", fmt_seat_range(summary_row.get("dem_seats_p25"), summary_row.get("dem_seats_p75")))
    c5.metric("National Environment", fmt_margin(summary_row.get("national_environment_margin")))
    c6.metric("Days Out", fmt_num(summary_row.get("days_out"), 0))

    st.divider()

    st.subheader("Seat Distribution")

    if seat_distribution.empty:
        st.info("No seat distribution file found.")
    else:
        sd = seat_distribution.copy()

        # Try common column names
        x_col = None
        y_col = None

        for col in ["dem_seats", "seats", "Democratic seats"]:
            if col in sd.columns:
                x_col = col
                break

        for col in ["probability", "share", "frequency"]:
            if col in sd.columns:
                y_col = col
                break

        if x_col and y_col:
            sd = sd.copy()
            sd["Control"] = sd[x_col].apply(
                lambda x: "Democratic Senate" if float(x) >= SENATE_CONTROL_THRESHOLD else "Republican Senate"
            )

            fig = px.bar(
                sd,
                x=x_col,
                y=y_col,
                color="Control",
                color_discrete_map={
                    "Democratic Senate": DEM_COLOR,
                    "Republican Senate": GOP_COLOR,
                },
                labels={
                    x_col: "Democratic seats",
                    y_col: "Probability",
                },
                title="Simulated Democratic Seat Distribution",
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Most Competitive Races")

    comp = race_stats.copy()
    comp["simulated_dem_win_prob"] = pd.to_numeric(
        comp.get("simulated_dem_win_prob"),
        errors="coerce"
    )
    comp["competitiveness"] = (comp["simulated_dem_win_prob"] - 0.5).abs()
    comp = comp.sort_values("competitiveness").head(12)

    display = []
    for _, row in comp.iterrows():
        display.append(
            {
                "State": row.get("state", ""),
                "Rating": race_rating_from_prob(row.get("simulated_dem_win_prob")),
                "Dem candidate": row.get("dem_candidate", ""),
                "GOP candidate": row.get("gop_candidate", ""),
                "Dem odds": fmt_pct(row.get("simulated_dem_win_prob")),
                "Model margin": fmt_margin(row.get("model_margin_dem")),
                "Middle 50% margin": fmt_margin_range(row.get("margin_p25_dem"), row.get("margin_p75_dem")),
                "Avg sim margin": fmt_margin(row.get("avg_simulated_margin_dem")),
            }
        )

    st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)


    st.divider()
    st.subheader("Model Odds Over Time")

    if forecast_history.empty:
        st.info("No forecast history yet. Run the Senate full pipeline to start building the time series.")
    else:
        history = forecast_history.copy()

        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
            history = history.dropna(subset=["timestamp"]).sort_values("timestamp")
            history["Run"] = history["timestamp"].dt.strftime("%b %d, %I:%M %p")
        elif "run_date" in history.columns:
            history["Run"] = history["run_date"].astype(str)
        else:
            history["Run"] = range(1, len(history) + 1)

        if "dem_control_probability" in history.columns:
            history["Dem control odds"] = pd.to_numeric(
                history["dem_control_probability"],
                errors="coerce",
            ) * 100

            fig_history = px.line(
                history,
                x="Run",
                y="Dem control odds",
                markers=True,
                labels={
                    "Run": "Run",
                    "Dem control odds": "Democratic Senate control odds (%)",
                },
                title="Senate Democratic Control Odds Over Time",
            )
            fig_history.update_layout(yaxis_ticksuffix="%", yaxis_range=[0, 100])
            st.plotly_chart(fig_history, use_container_width=True)
        else:
            st.info("Forecast history exists, but no dem_control_probability column was found.")

        with st.expander("Forecast history table"):
            display_cols = [
                "timestamp",
                "days_out",
                "expected_dem_seats",
                "median_dem_seats",
                "dem_control_probability",
                "national_environment_margin",
                "polling_weight",
                "fundamentals_weight",
                "total_error_sd",
            ]
            display_cols = [c for c in display_cols if c in history.columns]
            st.dataframe(history[display_cols].tail(25), use_container_width=True, hide_index=True)


# -----------------------------
# Race Ratings
# -----------------------------
with tab_races:
    st.subheader("Race Ratings")

    if race_stats.empty:
        st.info("No race stats found.")
    else:
        rs = race_stats.copy()

        rs["simulated_dem_win_prob_num"] = pd.to_numeric(
            rs.get("simulated_dem_win_prob"),
            errors="coerce"
        )
        rs["model_margin_dem_num"] = pd.to_numeric(
            rs.get("model_margin_dem"),
            errors="coerce"
        )
        rs["rating"] = rs["simulated_dem_win_prob_num"].apply(race_rating_from_prob)
        rs["competitiveness"] = (rs["simulated_dem_win_prob_num"] - 0.5).abs()

        view_mode = st.radio(
            "Sort races by",
            ["Competitiveness", "Dem win probability", "State"],
            horizontal=True,
        )

        if view_mode == "Competitiveness":
            rs = rs.sort_values("competitiveness")
        elif view_mode == "Dem win probability":
            rs = rs.sort_values("simulated_dem_win_prob_num", ascending=False)
        else:
            rs = rs.sort_values("state")

        chart_df = rs.copy()
        chart_df["Dem win probability"] = chart_df["simulated_dem_win_prob_num"]
        chart_df["State"] = chart_df["state"]

        chart_df["Favored Party"] = chart_df["Dem win probability"].apply(
            lambda p: "Democrat" if float(p) >= 0.5 else "Republican"
        )

        fig = px.bar(
            chart_df,
            x="Dem win probability",
            y="State",
            orientation="h",
            color="Favored Party",
            color_discrete_map={
                "Democrat": DEM_COLOR,
                "Republican": GOP_COLOR,
            },
            hover_data=[
                "dem_candidate",
                "gop_candidate",
                "rating",
                "model_margin_dem_num",
            ],
            title="Democratic Win Probability by Race",
        )
        fig.update_layout(xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        table_rows = []
        for _, row in rs.iterrows():
            table_rows.append(
                {
                    "State": row.get("state", ""),
                    "Rating": row.get("rating", ""),
                    "Dem candidate": row.get("dem_candidate", ""),
                    "GOP candidate": row.get("gop_candidate", ""),
                    "Holder": row.get("current_holder", ""),
                    "Dem odds": fmt_pct(row.get("simulated_dem_win_prob")),
                    "Model margin": fmt_margin(row.get("model_margin_dem")),
                    "Middle 50% margin": fmt_margin_range(row.get("margin_p25_dem"), row.get("margin_p75_dem")),
                    "Fundamentals": fmt_margin(row.get("fundamentals_margin_dem")),
                    "Polling": fmt_margin(row.get("polling_margin_dem")),
                    "Tipping share": fmt_pct(row.get("tipping_share_of_control_sims")),
                }
            )

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# -----------------------------
# Model Drivers
# -----------------------------
with tab_drivers:
    st.subheader("National Environment")

    if national_env.empty:
        st.info("No national environment file found.")
    else:
        env = national_env.iloc[-1]

        c1, c2 = st.columns(2)
        c1.metric(
            "Generic Ballot",
            fmt_margin(
                env.get("generic_ballot_margin_dem")
            ),
        )
        c2.metric(
            "National Environment",
            fmt_margin(
                env.get(
                    "national_environment_margin_dem"
                )
            ),
        )

        st.caption(
            "Current formula: 0.90 × generic ballot. "
            "Presidential approval and a standalone "
            "midterm adjustment are not used."
        )

        env_display_cols = [
            "as_of_date",
            "generic_ballot_margin_dem",
            "national_environment_margin_dem",
            "source_notes",
        ]
        env_display_cols = [c for c in env_display_cols if c in national_env.columns]

        with st.expander("Raw national environment inputs", expanded=False):
            st.dataframe(national_env[env_display_cols].tail(1), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Fundamentals and Polling Audit")

    if race_inputs.empty:
        st.info("No race inputs found.")
    else:
        audit = race_inputs.copy()

        if not race_stats.empty:
            stats_cols = [
                c for c in [
                    "state",
                    "model_margin_dem",
                    "simulated_dem_win_prob",
                    "avg_simulated_margin_dem",
                    "pre_sim_dem_win_prob",
                ]
                if c in race_stats.columns
            ]
            audit = audit.merge(race_stats[stats_cols], on="state", how="left")

        if not polling.empty:
            polling_cols = [
                c for c in [
                    "state",
                    "polling_margin_dem",
                    "poll_count",
                    "effective_poll_count",
                    "latest_poll_end_date",
                    "avg_poll_age_days",
                    "total_poll_weight",
                    "largest_pollster_weight_share",
                    "only_partisan_or_internal_polls",
                ]
                if c in polling.columns
            ]
            poll_view = polling[polling_cols].rename(
                columns={
                    "polling_margin_dem": "manual_polling_margin_dem",
                    "poll_count": "manual_poll_count",
                    "effective_poll_count": "manual_effective_poll_count",
                    "latest_poll_end_date": "manual_latest_poll_end_date",
                    "avg_poll_age_days": "manual_avg_poll_age_days",
                    "total_poll_weight": "manual_total_poll_weight",
                    "largest_pollster_weight_share": "manual_largest_pollster_weight_share",
                    "only_partisan_or_internal_polls": "manual_only_partisan_or_internal_polls",
                }
            )
            audit = audit.merge(poll_view, on="state", how="left")

        if not bayes.empty:
            bayes_cols = [
                c for c in [
                    "state",
                    "original_bayesian_polling_weight",
                    "cycle_max_polling_weight",
                    "poll_count_weight_multiplier",
                    "bayesian_polling_weight_capped_before_polling_confidence_accelerator",
                    "recent_poll_count_45d",
                    "most_recent_poll_end_date",
                    "polling_confidence_boost",
                    "polling_confidence_absolute_cap",
                    "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
                    "polling_confidence_weight_change",
                    "polling_confidence_margin_change_dem",
                    "bayesian_model_margin_dem_capped_before_polling_confidence_accelerator",
                    "bayesian_model_margin_dem_capped",
                    "bayesian_posterior_sd_calibrated",
                ]
                if c in bayes.columns
            ]
            audit = audit.merge(
                bayes[bayes_cols],
                on="state",
                how="left",
            )

        key_default = ["AK", "FL", "GA", "ME", "NC", "OH", "TX"]
        all_states = sorted(audit["state"].dropna().unique().tolist())
        selected = st.multiselect(
            "States to show",
            all_states,
            default=[s for s in key_default if s in all_states],
        )

        audit = audit[audit["state"].isin(selected)].copy()

        rows = []
        for _, row in audit.iterrows():
            rows.append(
                {
                    "State": row.get("state", ""),
                    "Baseline source": row.get("baseline_source", ""),
                    "Pres. baseline": fmt_margin(row.get("state_partisan_baseline_dem")),
                    "Nat'l env. effect": fmt_margin(row.get("state_environment_adjustment_dem")),
                    "Incumbency": fmt_margin(row.get("incumbency_adjustment_dem")),
                    "Cand. quality": fmt_margin(row.get("candidate_quality_adjustment_dem")),
                    "Manual CQ": fmt_margin(row.get("manual_candidate_quality_adjustment_dem")),
                    "Objective CQ": fmt_margin(row.get("objective_candidate_quality_adjustment_dem")),
                    "CQ gate": fmt_num(row.get("candidate_quality_gate"), 2),
                    "Prior elected": fmt_margin(row.get("prior_elected_experience_adjustment_dem")),
                    "Statewide win": fmt_margin(row.get("prior_statewide_win_adjustment_dem")),
                    "Overperf.": fmt_margin(row.get("overperformance_adjustment_dem")),
                    "Liability": fmt_margin(row.get("candidate_liability_adjustment_dem")),
                    "Special": fmt_margin(row.get("special_adjustment_dem")),
                    "Fundamentals": fmt_margin(row.get("fundamentals_margin_dem")),
                    "Manual polling": fmt_margin(row.get("manual_polling_margin_dem")),
                    "Poll count": fmt_num(row.get("manual_poll_count"), 0),
                    "Effective polls": fmt_num(row.get("manual_effective_poll_count"), 2),
                    "Recent polls": fmt_num(row.get("recent_poll_count_45d"), 0),
                    "Avg. poll age": fmt_num(row.get("manual_avg_poll_age_days"), 1),
                    "Largest pollster share": fmt_pct(row.get("manual_largest_pollster_weight_share")),
                    "Bayes margin": fmt_margin(row.get("bayesian_model_margin_dem")),
                    "Raw poll weight": fmt_pct(row.get("original_bayesian_polling_weight")),
                    "Cycle cap": fmt_pct(row.get("cycle_max_polling_weight")),
                    "Pre-boost weight": fmt_pct(
                        row.get(
                            "bayesian_polling_weight_capped_before_polling_confidence_accelerator"
                        )
                    ),
                    "Confidence boost": fmt_pct(row.get("polling_confidence_boost")),
                    "Final poll weight": fmt_pct(
                        row.get(
                            "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
                            row.get("bayesian_polling_weight"),
                        )
                    ),
                    "Boost margin effect": fmt_margin(
                        row.get("polling_confidence_margin_change_dem")
                    ),
                    "Posterior SD": fmt_num(
                        row.get(
                            "bayesian_posterior_sd_calibrated",
                            row.get("bayesian_posterior_sd"),
                        ),
                        2,
                    ),
                    "Final margin": fmt_margin(row.get("model_margin_dem")),
                    "Dem odds": fmt_pct(row.get("simulated_dem_win_prob")),
                    "Notes": row.get("fundamentals_notes", ""),
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("Full race input table", expanded=False):
            st.dataframe(race_inputs, use_container_width=True, hide_index=True)

# -----------------------------
# Scenarios
# -----------------------------
with tab_scenarios:
    st.subheader("National Environment Scenario Sensitivity")

    if scenarios.empty:
        st.info("No scenario summary found. Run `python3 scenario_runner.py`.")
    else:
        sc = scenarios.copy()
        sc["national_environment_margin_dem_num"] = pd.to_numeric(
            sc.get("national_environment_margin_dem"),
            errors="coerce"
        )
        sc["dem_control_probability_num"] = pd.to_numeric(
            sc.get("dem_control_probability"),
            errors="coerce"
        )
        sc["expected_dem_seats_num"] = pd.to_numeric(
            sc.get("expected_dem_seats"),
            errors="coerce"
        )
        sc = sc.sort_values("national_environment_margin_dem_num")

        fig = px.line(
            sc,
            x="national_environment_margin_dem_num",
            y="dem_control_probability_num",
            markers=True,
            hover_data=["scenario", "expected_dem_seats_num"],
            labels={
                "national_environment_margin_dem_num": "National environment margin",
                "dem_control_probability_num": "Democratic control probability",
            },
            title="Control Probability by National Environment",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        scen_rows = []
        for _, row in sc.iterrows():
            scen_rows.append(
                {
                    "Scenario": row.get("scenario", ""),
                    "National env.": fmt_margin(row.get("national_environment_margin_dem")),
                    "Shift from base": fmt_margin(row.get("environment_shift_from_base")),
                    "Dem control odds": fmt_pct(row.get("dem_control_probability")),
                    "Expected Dem seats": fmt_num(row.get("expected_dem_seats"), 2),
                    "Median Dem seats": fmt_num(row.get("median_dem_seats"), 0),
                }
            )
        st.dataframe(pd.DataFrame(scen_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("National Environment Formula Tests")

    formula_summary = read_csv_safe(OUTPUTS / "national_environment_formula_readable_summary.csv")

    if formula_summary.empty:
        st.info("No readable formula test found. Run `python3 summarize_formula_test.py`.")
    else:
        st.dataframe(formula_summary, use_container_width=True, hide_index=True)

# -----------------------------
# Diagnostics
# -----------------------------
with tab_diagnostics:
    st.subheader("Model Diagnostics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Error SD", fmt_num(summary_row.get("total_error_sd"), 2))
    c2.metric("National Error SD", fmt_num(summary_row.get("national_error_sd"), 2))
    c3.metric("Race Error SD", fmt_num(summary_row.get("race_error_sd"), 2))
    c4.metric("Implied Correlation", fmt_pct(summary_row.get("implied_correlation")))

    st.divider()

    st.subheader("File Status")

    files = [
        INPUTS / "race_inputs.csv",
        INPUTS / "national_environment.csv",
        INPUTS / "manual_polls.csv",
        OUTPUTS / "manual_polls_clean.csv",
        INPUTS / "polling_averages_generated.csv",
        INPUTS / "bayesian_update_generated.csv",
        OUTPUTS / "race_stats.csv",
        OUTPUTS / "forecast_summary.csv",
        OUTPUTS / "scenario_summary.csv",
    ]

    status_rows = []
    for f in files:
        status_rows.append(
            {
                "File": str(f),
                "Exists": f.exists(),
                "Size KB": fmt_num(f.stat().st_size / 1024, 1) if f.exists() else "—",
            }
        )

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Raw Outputs")

    with st.expander("forecast_summary.csv"):
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("race_stats.csv"):
        st.dataframe(race_stats, use_container_width=True, hide_index=True)

    with st.expander("bayesian_update_generated.csv"):
        st.dataframe(bayes, use_container_width=True, hide_index=True)


# -----------------------------
# Manual Poll Entry
# -----------------------------
with tab_manual_polls:
    render_candidate_event_registry_editor(
        default_chamber="senate",
        registry_path=(
            SHARED_MODEL_ROOT
            / "inputs"
            / "candidate_event_registry.csv"
        ),
        house_race_path=Path(
            "/Users/benyelin/Developer/"
            "house_model_python/inputs/"
            "house_race_inputs.csv"
        ),
        senate_race_path=(
            INPUTS
            / "race_inputs.csv"
        ),
        house_root=Path(
            "/Users/benyelin/Developer/"
            "house_model_python"
        ),
        senate_root=Path.cwd(),
        key_prefix="senate",
    )

    st.divider()
    st.subheader("Pollster House Effects")

    st.caption(
        "Maintain pollster-wide house effects separately from partisan or "
        "internal-poll adjustments. Positive values mean a pollster appears "
        "too Democratic and therefore reduce its reported Democratic margin. "
        "Negative values mean a pollster appears too Republican."
    )

    st.info(
        "Adjusted Democratic margin = reported Democratic margin − "
        "pollster house effect. Use Preview Changes before applying edits."
    )

    pollster_registry_path = INPUTS / "pollster_registry.csv"
    clean_manual_poll_path = OUTPUTS / "manual_polls_clean.csv"

    pollster_registry = read_csv_safe(
        pollster_registry_path
    )
    clean_manual_polls = read_csv_safe(
        clean_manual_poll_path
    )

    if pollster_registry.empty:
        st.warning(
            "No pollster registry was found at "
            f"{pollster_registry_path}."
        )
    else:
        registry_editor = pollster_registry.copy()

        required_registry_columns = {
            "canonical_pollster": "",
            "normalized_pollster_key": "",
            "aliases": "",
            "pollster_house_effect_dem": 0.0,
            "house_effect_confidence": "low",
            "house_effect_notes": "",
            "active": True,
        }

        for column, default in required_registry_columns.items():
            if column not in registry_editor.columns:
                registry_editor[column] = default

        registry_editor[
            "pollster_house_effect_dem"
        ] = pd.to_numeric(
            registry_editor[
                "pollster_house_effect_dem"
            ],
            errors="coerce",
        ).fillna(0.0)

        registry_editor[
            "house_effect_confidence"
        ] = (
            registry_editor[
                "house_effect_confidence"
            ]
            .fillna("low")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        registry_editor["house_effect_notes"] = (
            registry_editor["house_effect_notes"]
            .fillna("")
            .astype(str)
        )

        registry_editor["active"] = (
            registry_editor["active"]
            .fillna(True)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

        # Add read-only coverage information from the current clean
        # manual-poll database.
        coverage = pd.DataFrame(
            columns=[
                "canonical_pollster",
                "matching_poll_count",
                "matching_state_count",
                "matching_states",
            ]
        )

        if (
            not clean_manual_polls.empty
            and "canonical_pollster"
            in clean_manual_polls.columns
        ):
            coverage_source = clean_manual_polls.copy()

            coverage_source["canonical_pollster"] = (
                coverage_source["canonical_pollster"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            coverage_source["state"] = (
                coverage_source.get(
                    "state",
                    pd.Series(
                        "",
                        index=coverage_source.index,
                    ),
                )
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

            coverage = (
                coverage_source.loc[
                    coverage_source[
                        "canonical_pollster"
                    ].ne("")
                ]
                .groupby(
                    "canonical_pollster",
                    as_index=False,
                )
                .agg(
                    matching_poll_count=(
                        "canonical_pollster",
                        "size",
                    ),
                    matching_state_count=(
                        "state",
                        lambda values: values[
                            values.ne("")
                        ].nunique(),
                    ),
                    matching_states=(
                        "state",
                        lambda values: ", ".join(
                            sorted(
                                set(
                                    value
                                    for value in values
                                    if value
                                )
                            )
                        ),
                    ),
                )
            )

        registry_editor = registry_editor.merge(
            coverage,
            on="canonical_pollster",
            how="left",
        )

        registry_editor[
            "matching_poll_count"
        ] = pd.to_numeric(
            registry_editor.get(
                "matching_poll_count",
                0,
            ),
            errors="coerce",
        ).fillna(0).astype(int)

        registry_editor[
            "matching_state_count"
        ] = pd.to_numeric(
            registry_editor.get(
                "matching_state_count",
                0,
            ),
            errors="coerce",
        ).fillna(0).astype(int)

        registry_editor["matching_states"] = (
            registry_editor.get(
                "matching_states",
                pd.Series(
                    "",
                    index=registry_editor.index,
                ),
            )
            .fillna("")
            .astype(str)
        )

        registry_column_order = [
            "canonical_pollster",
            "aliases",
            "pollster_house_effect_dem",
            "house_effect_confidence",
            "house_effect_notes",
            "active",
            "matching_poll_count",
            "matching_state_count",
            "matching_states",
        ]

        edited_pollster_registry = st.data_editor(
            registry_editor[
                registry_column_order
            ],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="pollster_house_effect_registry_editor_v1",
            column_config={
                "canonical_pollster": (
                    st.column_config.TextColumn(
                        "Canonical pollster",
                        disabled=True,
                    )
                ),
                "aliases": st.column_config.TextColumn(
                    "Aliases",
                    disabled=True,
                ),
                "pollster_house_effect_dem": (
                    st.column_config.NumberColumn(
                        "House effect",
                        help=(
                            "Positive values reduce the "
                            "Democratic margin; negative "
                            "values increase it."
                        ),
                        step=0.1,
                        format="%.1f",
                    )
                ),
                "house_effect_confidence": (
                    st.column_config.SelectboxColumn(
                        "Confidence",
                        options=[
                            "low",
                            "medium",
                            "high",
                        ],
                    )
                ),
                "house_effect_notes": (
                    st.column_config.TextColumn(
                        "Rationale / notes",
                    )
                ),
                "active": st.column_config.CheckboxColumn(
                    "Active",
                    default=True,
                ),
                "matching_poll_count": (
                    st.column_config.NumberColumn(
                        "Polls",
                        disabled=True,
                        format="%d",
                    )
                ),
                "matching_state_count": (
                    st.column_config.NumberColumn(
                        "States",
                        disabled=True,
                        format="%d",
                    )
                ),
                "matching_states": (
                    st.column_config.TextColumn(
                        "Affected states",
                        disabled=True,
                    )
                ),
            },
        )

        current_registry_for_compare = (
            registry_editor[
                registry_column_order
            ]
            .copy()
            .set_index("canonical_pollster")
        )

        proposed_registry_for_compare = (
            edited_pollster_registry.copy()
            .set_index("canonical_pollster")
        )

        current_effect = pd.to_numeric(
            current_registry_for_compare[
                "pollster_house_effect_dem"
            ],
            errors="coerce",
        ).fillna(0.0)

        proposed_effect = pd.to_numeric(
            proposed_registry_for_compare[
                "pollster_house_effect_dem"
            ],
            errors="coerce",
        ).fillna(0.0)

        changed_effect = (
            proposed_effect - current_effect
        ).abs().gt(1e-12)

        current_confidence = (
            current_registry_for_compare[
                "house_effect_confidence"
            ]
            .fillna("")
            .astype(str)
        )

        proposed_confidence = (
            proposed_registry_for_compare[
                "house_effect_confidence"
            ]
            .fillna("")
            .astype(str)
        )

        current_notes = (
            current_registry_for_compare[
                "house_effect_notes"
            ]
            .fillna("")
            .astype(str)
        )

        proposed_notes = (
            proposed_registry_for_compare[
                "house_effect_notes"
            ]
            .fillna("")
            .astype(str)
        )

        current_active = (
            current_registry_for_compare[
                "active"
            ].astype(bool)
        )

        proposed_active = (
            proposed_registry_for_compare[
                "active"
            ].astype(bool)
        )

        changed_metadata = (
            current_confidence.ne(
                proposed_confidence
            )
            | current_notes.ne(proposed_notes)
            | current_active.ne(proposed_active)
        )

        changed_pollsters = sorted(
            set(
                proposed_registry_for_compare.index[
                    changed_effect | changed_metadata
                ].tolist()
            )
        )

        if changed_pollsters:
            st.caption(
                f"Unsaved changes detected for "
                f"{len(changed_pollsters)} pollster(s)."
            )
        else:
            st.caption(
                "No unsaved pollster-registry changes."
            )

        preview_col, apply_col = st.columns(
            [1, 1]
        )

        with preview_col:
            preview_house_effects = st.button(
                "Preview Changes",
                key=(
                    "preview_pollster_house_effects_v1"
                ),
            )

        with apply_col:
            apply_house_effects = st.button(
                "Apply and Rebuild Polling Averages",
                type="primary",
                key=(
                    "apply_pollster_house_effects_v1"
                ),
            )

        if preview_house_effects:
            if not changed_pollsters:
                st.info(
                    "No registry changes are currently "
                    "available to preview."
                )
            elif clean_manual_polls.empty:
                st.warning(
                    "The clean manual-poll file is unavailable, "
                    "so affected polling averages cannot be "
                    "previewed."
                )
            else:
                proposed_registry = (
                    pollster_registry.copy()
                )

                editable_columns = [
                    "canonical_pollster",
                    "pollster_house_effect_dem",
                    "house_effect_confidence",
                    "house_effect_notes",
                    "active",
                ]

                proposed_updates = (
                    edited_pollster_registry[
                        editable_columns
                    ]
                    .copy()
                    .set_index(
                        "canonical_pollster"
                    )
                )

                proposed_registry = (
                    proposed_registry
                    .set_index(
                        "canonical_pollster"
                    )
                )

                # CSV columns containing only blanks may be inferred as
                # float64. Normalize their dtypes before assigning edited
                # text and Boolean values from the Streamlit editor.
                proposed_registry[
                    "pollster_house_effect_dem"
                ] = pd.to_numeric(
                    proposed_registry[
                        "pollster_house_effect_dem"
                    ],
                    errors="coerce",
                ).fillna(0.0)

                for text_column in [
                    "house_effect_confidence",
                    "house_effect_notes",
                ]:
                    proposed_registry[text_column] = (
                        proposed_registry[text_column]
                        .fillna("")
                        .astype(str)
                    )

                proposed_registry["active"] = (
                    proposed_registry["active"]
                    .fillna(True)
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes", "y"])
                )

                for column in [
                    "pollster_house_effect_dem",
                    "house_effect_confidence",
                    "house_effect_notes",
                    "active",
                ]:
                    proposed_registry.loc[
                        proposed_updates.index,
                        column,
                    ] = proposed_updates[column]

                proposed_registry = (
                    proposed_registry.reset_index()
                )

                temporary_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".csv",
                        delete=False,
                    ) as temporary_file:
                        temporary_path = Path(
                            temporary_file.name
                        )
                        proposed_registry.to_csv(
                            temporary_file,
                            index=False,
                        )

                    preview_polls = (
                        apply_pollster_registry(
                            clean_manual_polls,
                            registry_path=temporary_path,
                        )
                    )

                    proposed_registry_effect = (
                        pd.to_numeric(
                            preview_polls[
                                "pollster_house_effect_dem"
                            ],
                            errors="coerce",
                        ).fillna(0.0)
                    )

                    override = pd.to_numeric(
                        preview_polls.get(
                            "manual_house_effect_override_dem",
                            pd.Series(
                                np.nan,
                                index=preview_polls.index,
                            ),
                        ),
                        errors="coerce",
                    )

                    preview_polls[
                        "preview_effective_house_effect_dem"
                    ] = override.where(
                        override.notna(),
                        proposed_registry_effect,
                    )

                    preview_polls[
                        "preview_adjusted_margin_dem"
                    ] = (
                        pd.to_numeric(
                            preview_polls[
                                "reported_margin_dem"
                            ],
                            errors="coerce",
                        )
                        - preview_polls[
                            "preview_effective_house_effect_dem"
                        ]
                    )

                    preview_polls[
                        "current_adjusted_margin_dem"
                    ] = pd.to_numeric(
                        preview_polls.get(
                            "final_poll_margin_dem",
                            np.nan,
                        ),
                        errors="coerce",
                    )

                    preview_polls[
                        "poll_margin_change_dem"
                    ] = (
                        preview_polls[
                            "preview_adjusted_margin_dem"
                        ]
                        - preview_polls[
                            "current_adjusted_margin_dem"
                        ]
                    )

                    affected_polls = preview_polls.loc[
                        preview_polls[
                            "canonical_pollster"
                        ].isin(changed_pollsters)
                    ].copy()

                    st.markdown(
                        "#### Proposed Pollster Changes"
                    )

                    change_rows = []

                    for canonical in changed_pollsters:
                        current_row = (
                            current_registry_for_compare
                            .loc[canonical]
                        )
                        proposed_row = (
                            proposed_registry_for_compare
                            .loc[canonical]
                        )

                        matching = affected_polls.loc[
                            affected_polls[
                                "canonical_pollster"
                            ].eq(canonical)
                        ]

                        affected_states = sorted(
                            matching.get(
                                "state",
                                pd.Series(dtype=str),
                            )
                            .fillna("")
                            .astype(str)
                            .str.upper()
                            .loc[
                                lambda values: values.ne("")
                            ]
                            .unique()
                            .tolist()
                        )

                        change_rows.append(
                            {
                                "Pollster": canonical,
                                "Current effect": float(
                                    pd.to_numeric(
                                        current_row[
                                            "pollster_house_effect_dem"
                                        ],
                                        errors="coerce",
                                    )
                                ),
                                "Proposed effect": float(
                                    pd.to_numeric(
                                        proposed_row[
                                            "pollster_house_effect_dem"
                                        ],
                                        errors="coerce",
                                    )
                                ),
                                "Matching polls": int(
                                    len(matching)
                                ),
                                "Affected states": (
                                    ", ".join(
                                        affected_states
                                    )
                                ),
                                "Confidence": str(
                                    proposed_row[
                                        "house_effect_confidence"
                                    ]
                                ),
                                "Notes": str(
                                    proposed_row[
                                        "house_effect_notes"
                                    ]
                                ),
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(change_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if not affected_polls.empty:
                        affected_poll_columns = [
                            column
                            for column in [
                                "race",
                                "state",
                                "pollster_raw",
                                "canonical_pollster",
                                "start_date",
                                "end_date",
                                "reported_margin_dem",
                                "current_adjusted_margin_dem",
                                "preview_effective_house_effect_dem",
                                "preview_adjusted_margin_dem",
                                "poll_margin_change_dem",
                                "poll_weight",
                            ]
                            if column
                            in affected_polls.columns
                        ]

                        st.markdown(
                            "#### Affected Polls"
                        )

                        st.dataframe(
                            affected_polls[
                                affected_poll_columns
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

                    weight_column = next(
                        (
                            column
                            for column in [
                                "poll_weight",
                                "final_poll_weight",
                            ]
                            if column
                            in preview_polls.columns
                        ),
                        None,
                    )

                    if (
                        weight_column is not None
                        and "state"
                        in preview_polls.columns
                    ):
                        preview_polls[
                            "_preview_weight"
                        ] = pd.to_numeric(
                            preview_polls[
                                weight_column
                            ],
                            errors="coerce",
                        ).fillna(0.0)

                        state_rows = []

                        for state, state_frame in (
                            preview_polls.groupby(
                                "state",
                                sort=True,
                            )
                        ):
                            weights = state_frame[
                                "_preview_weight"
                            ]

                            valid_current = (
                                state_frame[
                                    "current_adjusted_margin_dem"
                                ].notna()
                                & weights.gt(0.0)
                            )

                            valid_preview = (
                                state_frame[
                                    "preview_adjusted_margin_dem"
                                ].notna()
                                & weights.gt(0.0)
                            )

                            if (
                                not valid_current.any()
                                or not valid_preview.any()
                            ):
                                continue

                            current_average = float(
                                np.average(
                                    state_frame.loc[
                                        valid_current,
                                        "current_adjusted_margin_dem",
                                    ],
                                    weights=weights.loc[
                                        valid_current
                                    ],
                                )
                            )

                            preview_average = float(
                                np.average(
                                    state_frame.loc[
                                        valid_preview,
                                        "preview_adjusted_margin_dem",
                                    ],
                                    weights=weights.loc[
                                        valid_preview
                                    ],
                                )
                            )

                            if (
                                abs(
                                    preview_average
                                    - current_average
                                )
                                <= 1e-12
                            ):
                                continue

                            state_rows.append(
                                {
                                    "State": state,
                                    "Current polling average": (
                                        current_average
                                    ),
                                    "Preview polling average": (
                                        preview_average
                                    ),
                                    "Change": (
                                        preview_average
                                        - current_average
                                    ),
                                }
                            )

                        st.markdown(
                            "#### Previewed State Polling Averages"
                        )

                        if state_rows:
                            state_preview = (
                                pd.DataFrame(state_rows)
                                .sort_values(
                                    "Change",
                                    key=lambda values: (
                                        values.abs()
                                    ),
                                    ascending=False,
                                )
                            )

                            st.dataframe(
                                state_preview,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info(
                                "The proposed edits do not "
                                "change any weighted state "
                                "polling average."
                            )
                    else:
                        st.warning(
                            "No poll-weight column was found, "
                            "so the dashboard cannot preview "
                            "weighted state averages."
                        )

                    st.success(
                        "Dry run complete. No files have "
                        "been changed."
                    )
                finally:
                    if (
                        temporary_path is not None
                        and temporary_path.exists()
                    ):
                        temporary_path.unlink()

        if apply_house_effects:
            if not changed_pollsters:
                st.info(
                    "No registry changes are currently "
                    "available to apply."
                )
            else:
                registry_to_save = (
                    pollster_registry.copy()
                    .set_index(
                        "canonical_pollster"
                    )
                )

                # Normalize registry dtypes before assigning edited
                # numbers, text, and Boolean values.
                registry_to_save[
                    "pollster_house_effect_dem"
                ] = pd.to_numeric(
                    registry_to_save[
                        "pollster_house_effect_dem"
                    ],
                    errors="coerce",
                ).fillna(0.0)

                for text_column in [
                    "house_effect_confidence",
                    "house_effect_notes",
                ]:
                    registry_to_save[text_column] = (
                        registry_to_save[text_column]
                        .fillna("")
                        .astype(str)
                    )

                registry_to_save["active"] = (
                    registry_to_save["active"]
                    .fillna(True)
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes", "y"])
                )

                proposed_updates = (
                    edited_pollster_registry[
                        [
                            "canonical_pollster",
                            "pollster_house_effect_dem",
                            "house_effect_confidence",
                            "house_effect_notes",
                            "active",
                        ]
                    ]
                    .copy()
                    .set_index(
                        "canonical_pollster"
                    )
                )

                for column in [
                    "pollster_house_effect_dem",
                    "house_effect_confidence",
                    "house_effect_notes",
                    "active",
                ]:
                    registry_to_save.loc[
                        proposed_updates.index,
                        column,
                    ] = proposed_updates[column]

                registry_to_save = (
                    registry_to_save.reset_index()
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                registry_backup = (
                    pollster_registry_path.with_name(
                        f"{pollster_registry_path.stem}."
                        f"before_dashboard_house_effect_"
                        f"{timestamp}"
                        f"{pollster_registry_path.suffix}"
                    )
                )

                pollster_registry.to_csv(
                    registry_backup,
                    index=False,
                )

                registry_to_save.to_csv(
                    pollster_registry_path,
                    index=False,
                )

                validation_result = subprocess.run(
                    [
                        sys.executable,
                        "validate_manual_polls.py",
                    ],
                    capture_output=True,
                    text=True,
                )

                if validation_result.returncode != 0:
                    pollster_registry.to_csv(
                        pollster_registry_path,
                        index=False,
                    )

                    st.error(
                        "Manual-poll validation failed. "
                        "The prior registry was restored."
                    )

                    st.code(
                        validation_result.stdout
                        + "\n"
                        + validation_result.stderr
                    )
                else:
                    ingestion_result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "senate_model.poll_ingestion",
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if ingestion_result.returncode != 0:
                        pollster_registry.to_csv(
                            pollster_registry_path,
                            index=False,
                        )

                        subprocess.run(
                            [
                                sys.executable,
                                "validate_manual_polls.py",
                            ],
                            capture_output=True,
                            text=True,
                        )

                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "senate_model.poll_ingestion",
                            ],
                            capture_output=True,
                            text=True,
                        )

                        st.error(
                            "Polling aggregation failed. "
                            "The prior registry and polling "
                            "outputs were restored."
                        )

                        st.code(
                            ingestion_result.stdout
                            + "\n"
                            + ingestion_result.stderr
                        )
                    else:
                        st.success(
                            "Pollster house effects were saved "
                            "and polling averages were rebuilt."
                        )

                        st.caption(
                            f"Registry backup: "
                            f"{registry_backup}"
                        )

                        with st.expander(
                            "Pipeline output"
                        ):
                            st.code(
                                validation_result.stdout
                                + "\n"
                                + ingestion_result.stdout
                            )

                        st.rerun()

    st.divider()
    st.subheader("Manual Poll Entry")

    st.caption(
        "Add, edit, or delete manually entered Senate polls. Partisan/sponsor metadata "
        "feeds the partisan pollster adjustment script. Manual house-effect fields are "
        "not exposed here; poll adjustments are generated by the pipeline."
    )

    manual_poll_path = INPUTS / "manual_polls.csv"

    manual_poll_columns = [
        "race",
        "state",
        "chamber",
        "pollster",
        "pollster_grade",
        "sponsor",
        "poll_sponsor_type",
        "partisan_sponsor_party",
        "is_internal_poll",
        "pollster_partisan_affiliation",
        "partisan_pollster_review_notes",
        "start_date",
        "end_date",
        "sample_size",
        "sample_type",
        "dem_candidate",
        "rep_candidate",
        "ind_candidate",
        "other_candidate",
        "dem_pct",
        "rep_pct",
        "ind_pct",
        "other_pct",
        "undecided_pct",
        "notes",
    ]

    numeric_poll_columns = [
        "sample_size",
        "dem_pct",
        "rep_pct",
        "ind_pct",
        "other_pct",
        "undecided_pct",
    ]

    text_metadata_cols = [
        "race",
        "state",
        "chamber",
        "pollster",
        "pollster_grade",
        "sponsor",
        "poll_sponsor_type",
        "partisan_sponsor_party",
        "pollster_partisan_affiliation",
        "partisan_pollster_review_notes",
        "sample_type",
        "dem_candidate",
        "rep_candidate",
        "ind_candidate",
        "other_candidate",
        "notes",
    ]

    existing_manual_polls = read_csv_safe(manual_poll_path)

    if existing_manual_polls.empty:
        existing_manual_polls = pd.DataFrame(columns=manual_poll_columns)

    # Ensure all approved manual-entry columns exist.
    for col in manual_poll_columns:
        if col not in existing_manual_polls.columns:
            if col == "is_internal_poll":
                existing_manual_polls[col] = False
            else:
                existing_manual_polls[col] = ""

    # Keep only approved manual-entry columns. Generated/audit columns should not be edited here.
    existing_manual_polls = existing_manual_polls.loc[:, manual_poll_columns].copy()

    # Normalize text columns for Streamlit editor compatibility.
    for col in text_metadata_cols:
        if col in existing_manual_polls.columns:
            existing_manual_polls[col] = (
                existing_manual_polls[col]
                .fillna("")
                .astype(str)
                .replace({"nan": "", "None": "", "NaN": ""})
            )

    # Normalize booleans.
    existing_manual_polls["is_internal_poll"] = (
        existing_manual_polls["is_internal_poll"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    # Normalize numeric columns.
    for col in numeric_poll_columns:
        if col in existing_manual_polls.columns:
            existing_manual_polls[col] = pd.to_numeric(
                existing_manual_polls[col],
                errors="coerce",
            )

    st.markdown("### Edit Existing Manual Polls")

    st.caption(
        "Use the table below to edit existing polls. To delete a poll, check Delete "
        "and then click Save Edits / Delete Marked Polls."
    )

    editable = existing_manual_polls.copy()
    editable.insert(0, "delete", False)
    editable.insert(1, "row_id", range(1, len(editable) + 1))

    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_order=["delete", "row_id"] + manual_poll_columns,
        key="manual_poll_editor_unified_v1",
        column_config={
            "delete": st.column_config.CheckboxColumn(
                "Delete",
                default=False,
            ),
            "row_id": st.column_config.NumberColumn(
                "Row",
                disabled=True,
            ),
            "pollster_grade": st.column_config.SelectboxColumn(
                "Pollster grade",
                options=["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "Unknown"],
            ),
            "sample_type": st.column_config.SelectboxColumn(
                "Sample type",
                options=["LV", "RV", "A", "Other"],
            ),
            "poll_sponsor_type": st.column_config.SelectboxColumn(
                "Sponsor type",
                options=["", "independent", "media", "university", "party", "campaign", "super PAC", "other"],
            ),
            "partisan_sponsor_party": st.column_config.SelectboxColumn(
                "Sponsor party",
                options=["", "D", "R", "none", "unknown"],
            ),
            "is_internal_poll": st.column_config.CheckboxColumn(
                "Internal/campaign poll",
                default=False,
            ),
            "pollster_partisan_affiliation": st.column_config.SelectboxColumn(
                "Pollster partisan affiliation",
                options=["", "D", "R", "none", "unknown"],
            ),
            "partisan_pollster_review_notes": st.column_config.TextColumn(
                "Partisan poll notes",
            ),
            "sample_size": st.column_config.NumberColumn(
                "Sample size",
                min_value=0,
                step=1,
                format="%d",
            ),
            "dem_pct": st.column_config.NumberColumn("Dem %", step=0.1, format="%.1f"),
            "rep_pct": st.column_config.NumberColumn("Rep %", step=0.1, format="%.1f"),
            "ind_pct": st.column_config.NumberColumn("Ind %", step=0.1, format="%.1f"),
            "other_pct": st.column_config.NumberColumn("Other %", step=0.1, format="%.1f"),
            "undecided_pct": st.column_config.NumberColumn("Undecided %", step=0.1, format="%.1f"),
        },
    )

    c_save, c_reset = st.columns([1, 3])

    with c_save:
        save_edits = st.button(
            "Save Edits / Delete Marked Polls",
            type="primary",
            key="save_manual_poll_edits_unified_v1",
        )

    with c_reset:
        st.caption("Saving will overwrite inputs/manual_polls.csv and create a .bak backup.")

    if save_edits:
        updated = edited.copy()

        if "delete" in updated.columns:
            updated = updated[~updated["delete"].fillna(False)].copy()

        for col in ["delete", "row_id"]:
            if col in updated.columns:
                updated = updated.drop(columns=[col])

        for col in manual_poll_columns:
            if col not in updated.columns:
                if col == "is_internal_poll":
                    updated[col] = False
                else:
                    updated[col] = ""

        updated = updated.loc[:, manual_poll_columns].copy()

        updated["state"] = updated["state"].fillna("").astype(str).str.strip().str.upper()

        for col in text_metadata_cols:
            if col in updated.columns:
                updated[col] = (
                    updated[col]
                    .fillna("")
                    .astype(str)
                    .replace({"nan": "", "None": "", "NaN": ""})
                )

        updated["is_internal_poll"] = (
            updated["is_internal_poll"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

        for col in numeric_poll_columns:
            if col in updated.columns:
                updated[col] = pd.to_numeric(updated[col], errors="coerce")

        nonblank_mask = updated[
            ["race", "state", "pollster", "dem_pct", "rep_pct"]
        ].notna().any(axis=1)
        updated = updated[nonblank_mask].copy()

        manual_poll_path.parent.mkdir(parents=True, exist_ok=True)

        if manual_poll_path.exists():
            backup_path = manual_poll_path.with_suffix(".csv.bak")
            existing_manual_polls.to_csv(backup_path, index=False)

        updated.to_csv(manual_poll_path, index=False)

        st.success(
            f"Saved {len(updated)} manual polls to {manual_poll_path}. "
            "Run the full pipeline to ingest the changes."
        )

    st.divider()

    st.markdown("### Add New Poll")

    with st.form("manual_poll_entry_form_unified_v1"):
        c1, c2, c3 = st.columns(3)

        with c1:
            race = st.text_input("Race", value="")
            state = st.text_input("State", value="")
            chamber = st.text_input("Chamber", value="Senate")
            pollster = st.text_input("Pollster", value="")
            pollster_grade = st.selectbox(
                "Pollster grade",
                ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "Unknown"],
                index=4,
            )
            sponsor = st.text_input("Sponsor", value="")

        with c2:
            start_date = st.date_input("Start date")
            end_date = st.date_input("End date")
            sample_size = st.number_input("Sample size", min_value=0, value=800, step=1)
            sample_type = st.selectbox("Sample type", ["LV", "RV", "A", "Other"], index=0)
            dem_candidate = st.text_input("Dem candidate", value="")
            rep_candidate = st.text_input("Rep candidate", value="")

        with c3:
            ind_candidate = st.text_input("Ind candidate", value="")
            other_candidate = st.text_input("Other candidate", value="")
            dem_pct = st.number_input("Dem %", value=0.0, step=0.1, format="%.1f")
            rep_pct = st.number_input("Rep %", value=0.0, step=0.1, format="%.1f")
            ind_pct = st.number_input("Ind %", value=0.0, step=0.1, format="%.1f")
            other_pct = st.number_input("Other %", value=0.0, step=0.1, format="%.1f")
            undecided_pct = st.number_input("Undecided %", value=0.0, step=0.1, format="%.1f")

        st.markdown("#### Partisan / Sponsor Metadata")

        p1, p2, p3 = st.columns(3)

        with p1:
            poll_sponsor_type = st.selectbox(
                "Sponsor type",
                ["", "independent", "media", "university", "party", "campaign", "super PAC", "other"],
                index=0,
            )

        with p2:
            partisan_sponsor_party = st.selectbox(
                "Sponsor party",
                ["", "D", "R", "none", "unknown"],
                index=0,
            )

        with p3:
            pollster_partisan_affiliation = st.selectbox(
                "Pollster partisan affiliation",
                ["", "D", "R", "none", "unknown"],
                index=0,
            )

        is_internal_poll = st.checkbox("Internal/campaign poll", value=False)

        partisan_pollster_review_notes = st.text_input("Partisan poll notes", value="")
        notes = st.text_area("General notes", value="")

        submitted = st.form_submit_button("Add Poll")

        if submitted:
            new_row = {
                "race": race,
                "state": state.strip().upper(),
                "chamber": chamber,
                "pollster": pollster,
                "pollster_grade": pollster_grade,
                "sponsor": sponsor,
                "poll_sponsor_type": poll_sponsor_type,
                "partisan_sponsor_party": partisan_sponsor_party,
                "is_internal_poll": is_internal_poll,
                "pollster_partisan_affiliation": pollster_partisan_affiliation,
                "partisan_pollster_review_notes": partisan_pollster_review_notes,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "sample_size": sample_size,
                "sample_type": sample_type,
                "dem_candidate": dem_candidate,
                "rep_candidate": rep_candidate,
                "ind_candidate": ind_candidate,
                "other_candidate": other_candidate,
                "dem_pct": dem_pct,
                "rep_pct": rep_pct,
                "ind_pct": ind_pct,
                "other_pct": other_pct,
                "undecided_pct": undecided_pct,
                "notes": notes,
            }

            updated = pd.concat(
                [
                    existing_manual_polls,
                    pd.DataFrame([new_row]),
                ],
                ignore_index=True,
            )

            for col in manual_poll_columns:
                if col not in updated.columns:
                    updated[col] = ""

            updated = updated.loc[:, manual_poll_columns].copy()

            manual_poll_path.parent.mkdir(parents=True, exist_ok=True)
            updated.to_csv(manual_poll_path, index=False)

            st.success(f"Saved new poll to {manual_poll_path}. Run the full pipeline to ingest it.")
            st.dataframe(updated.tail(10), use_container_width=True, hide_index=True)

    st.divider()

