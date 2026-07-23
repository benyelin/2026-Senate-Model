from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

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

