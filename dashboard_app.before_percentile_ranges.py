from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import subprocess

DEM_COLOR = "#1f77b4"
GOP_COLOR = "#d62728"
SENATE_CONTROL_THRESHOLD = 51


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
INPUTS = ROOT / "inputs"

st.set_page_config(page_title="2026 Senate Forecast", page_icon="🗳️", layout="wide")


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

def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "—"


def num(x, digits=1):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


summary = read_csv_safe(OUTPUTS / "forecast_summary.csv")
race_stats = read_csv_safe(OUTPUTS / "race_stats.csv")
seat_dist = read_csv_safe(OUTPUTS / "seat_distribution.csv")
scenarios = read_csv_safe(OUTPUTS / "scenario_summary.csv")
bayes = read_csv_safe(INPUTS / "bayesian_update_generated.csv")
poll_avgs = read_csv_safe(INPUTS / "polling_averages_generated.csv")

st.title("2026 Senate Forecast Dashboard")
st.caption("Polling ingestion → Bayesian update → correlated simulation → scenario diagnostics.")

if summary.empty:
    st.error("No forecast outputs found. Run `python run_full_pipeline.py` first.")
    st.stop()

s = summary.iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dem Control Odds", pct(s.get("dem_control_probability")))
c2.metric("Expected Dem Seats", num(s.get("expected_dem_seats"), 2))
c3.metric("Median Dem Seats", num(s.get("median_dem_seats"), 0))
c4.metric("Polling Weight", pct(s.get("polling_weight")))
c5.metric("Implied Correlation", pct(s.get("implied_correlation")))

c6, c7, c8, c9 = st.columns(4)
c6.metric("Days Out", num(s.get("days_out"), 0))
c7.metric("Total Error SD", num(s.get("total_error_sd"), 2))
c8.metric("National Error SD", num(s.get("national_error_sd"), 2))
c9.metric("Race Error SD", num(s.get("race_error_sd"), 2))

st.divider()

tab_overview, tab_races, tab_scenarios, tab_polls, tab_method = st.tabs(
    ["Overview", "Race Table", "Scenarios", "Polling & Bayes", "Method"]
)

with tab_overview:

    # -----------------------------
    # Compact Model Snapshot
    # -----------------------------
    st.markdown("### Model Snapshot")

    def _fmt_margin(x):
        try:
            x = float(x)
        except Exception:
            return "—"
        if x > 0:
            return f"D+{x:.1f}"
        if x < 0:
            return f"R+{abs(x):.1f}"
        return "Even"

    def _fmt_pct(x):
        try:
            return f"{float(x):.1%}"
        except Exception:
            return "—"

    def _fmt_num(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "—"

    def _fmt_signed(x):
        try:
            return f"{float(x):+.1f}"
        except Exception:
            return "—"

    snapshot_env = read_csv_safe(INPUTS / "national_environment.csv")
    snapshot_races = read_csv_safe(INPUTS / "race_inputs.csv")
    snapshot_stats = read_csv_safe(OUTPUTS / "race_stats.csv")
    snapshot_scenarios = read_csv_safe(OUTPUTS / "scenario_summary.csv")
    snapshot_polling = read_csv_safe(INPUTS / "polling_averages_generated.csv")
    snapshot_bayes = read_csv_safe(INPUTS / "bayesian_update_generated.csv")

    if not snapshot_env.empty:
        env = snapshot_env.iloc[-1]

        env_cols = st.columns(4)
        env_cols[0].metric(
            "Generic Ballot",
            _fmt_margin(env.get("generic_ballot_margin_dem"))
        )
        env_cols[1].metric(
            "GOP Pres. Net Approval",
            _fmt_signed(env.get("presidential_net_approval"))
        )
        env_cols[2].metric(
            "Approval Adj.",
            _fmt_margin(env.get("approval_adjustment_dem"))
        )
        env_cols[3].metric(
            "National Environment",
            _fmt_margin(env.get("national_environment_margin_dem"))
        )

        with st.expander("National environment details", expanded=False):
            show_cols = [
                "as_of_date",
                "generic_ballot_margin_dem",
                "presidential_approval",
                "presidential_disapproval",
                "presidential_net_approval",
                "approval_adjustment_dem",
                "midterm_adjustment_dem",
                "national_environment_margin_dem",
                "source_notes",
            ]
            show_cols = [c for c in show_cols if c in snapshot_env.columns]
            st.dataframe(snapshot_env[show_cols].tail(1), use_container_width=True, hide_index=True)

    st.markdown("#### Key Race Snapshot")

    if not snapshot_races.empty:
        races_view = snapshot_races.copy()

        if "state" in races_view.columns:
            races_view["state"] = races_view["state"].astype(str).str.strip().str.upper()

        if not snapshot_stats.empty and "state" in snapshot_stats.columns:
            stats_view = snapshot_stats.copy()
            stats_view["state"] = stats_view["state"].astype(str).str.strip().str.upper()

            stats_cols = [
                c for c in [
                    "state",
                    "model_margin_dem",
                    "simulated_dem_win_prob",
                    "avg_simulated_margin_dem",
                ]
                if c in stats_view.columns
            ]

            races_view = races_view.merge(
                stats_view[stats_cols],
                on="state",
                how="left"
            )

        if not snapshot_polling.empty and "state" in snapshot_polling.columns:
            polling_view = snapshot_polling.copy()
            polling_view["state"] = (
                polling_view["state"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            polling_cols = [
                c for c in [
                    "state",
                    "polling_margin_dem",
                    "poll_count",
                    "effective_poll_count",
                    "avg_poll_age_days",
                    "largest_pollster_weight_share",
                ]
                if c in polling_view.columns
            ]

            races_view = races_view.merge(
                polling_view[polling_cols],
                on="state",
                how="left",
                suffixes=("", "_polling_snapshot"),
            )

        if not snapshot_bayes.empty and "state" in snapshot_bayes.columns:
            bayes_view = snapshot_bayes.copy()
            bayes_view["state"] = (
                bayes_view["state"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            bayes_cols = [
                c for c in [
                    "state",
                    "recent_poll_count_45d",
                    "polling_confidence_boost",
                    "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
                ]
                if c in bayes_view.columns
            ]

            races_view = races_view.merge(
                bayes_view[bayes_cols],
                on="state",
                how="left",
            )

        default_key_states = ["AK", "FL", "GA", "ME", "NC", "OH", "TX"]
        key_states = [s for s in default_key_states if s in races_view["state"].values]

        races_view = races_view[races_view["state"].isin(key_states)].copy()

        compact_rows = []

        for _, row in races_view.iterrows():
            compact_rows.append(
                {
                    "State": row.get("state", ""),
                    "Race": f"{row.get('dem_candidate', '')} vs. {row.get('gop_candidate', '')}",
                    "Pres. baseline": _fmt_margin(row.get("state_partisan_baseline_dem")),
                    "Nat'l env. effect": _fmt_margin(row.get("state_environment_adjustment_dem")),
                    "Incumbency": _fmt_margin(row.get("incumbency_adjustment_dem")),
                    "Cand. quality": _fmt_margin(row.get("candidate_quality_adjustment_dem")),
                    "Fundamentals": _fmt_margin(row.get("fundamentals_margin_dem")),
                    "Bayes margin": _fmt_margin(row.get("bayesian_model_margin_dem")),
                    "Final margin": _fmt_margin(row.get("model_margin_dem")),
                    "Dem win odds": _fmt_pct(row.get("simulated_dem_win_prob")),
                    "Poll avg.": _fmt_margin(row.get("polling_margin_dem")),
                    "Polls": _fmt_num(row.get("poll_count")),
                    "Effective polls": _fmt_num(row.get("effective_poll_count")),
                    "Recent polls": _fmt_num(row.get("recent_poll_count_45d")),
                    "Poll weight": _fmt_pct(
                        row.get(
                            "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
                            row.get("bayesian_polling_weight"),
                        )
                    ),
                    "Confidence boost": _fmt_pct(row.get("polling_confidence_boost")),
                    "Avg. poll age": _fmt_num(row.get("avg_poll_age_days")),
                    "Uncertainty": _fmt_num(row.get("bayesian_posterior_sd")),
                }
            )

        if compact_rows:
            st.dataframe(
                pd.DataFrame(compact_rows),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No key race snapshot available yet.")

    st.markdown("#### Scenario Sensitivity")

    if not snapshot_scenarios.empty:
        scen = snapshot_scenarios.copy()

        scenario_rows = []
        for _, row in scen.iterrows():
            scenario_rows.append(
                {
                    "Scenario": row.get("scenario", ""),
                    "National env.": _fmt_margin(row.get("national_environment_margin_dem")),
                    "Shift": _fmt_margin(row.get("environment_shift_from_base")),
                    "Dem control odds": _fmt_pct(row.get("dem_control_probability")),
                    "Expected Dem seats": _fmt_num(row.get("expected_dem_seats")),
                    "Median Dem seats": _fmt_num(row.get("median_dem_seats")),
                }
            )

        st.dataframe(
            pd.DataFrame(scenario_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No scenario summary available. Run `python3 scenario_runner.py`.")

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Seat Distribution")
        if seat_dist.empty:
            st.info("No seat distribution file found.")
        else:
            seat_dist = seat_dist.copy()
            seat_dist["Control"] = seat_dist["dem_seats"].apply(
                lambda x: "Democratic Senate" if x >= SENATE_CONTROL_THRESHOLD else "Republican Senate"
            )

            fig = px.bar(
                seat_dist,
                x="dem_seats",
                y="probability",
                color="Control",
                color_discrete_map={
                    "Democratic Senate": DEM_COLOR,
                    "Republican Senate": GOP_COLOR,
                },
                labels={"dem_seats": "Democratic Seats", "probability": "Probability"},
                text=seat_dist["probability"].map(lambda v: f"{v:.1%}"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_tickformat=".0%", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model Settings")
        st.write(f"Control threshold: **{int(s.get('dem_control_threshold', 51))} seats**")
        st.write(f"Baseline Democratic seats: **{int(s.get('dem_baseline_seats', 47))}**")
        st.write(f"National environment margin: **{num(s.get('national_environment_margin'), 2)}**")

        if not race_stats.empty and "tipping_share_of_control_sims" in race_stats.columns:
            tipping = race_stats.sort_values("tipping_share_of_control_sims", ascending=False).head(8)
            show = tipping[["state", "tipping_share_of_control_sims", "simulated_dem_win_prob", "model_margin_dem"]].copy()
            show["tipping_share_of_control_sims"] = show["tipping_share_of_control_sims"].map(lambda v: f"{v:.1%}")
            show["simulated_dem_win_prob"] = show["simulated_dem_win_prob"].map(lambda v: f"{v:.1%}")
            show["model_margin_dem"] = show["model_margin_dem"].map(lambda v: f"{v:.1f}")
            st.subheader("Most Common Tipping Races")
            st.dataframe(show.rename(columns={
                "state": "State",
                "tipping_share_of_control_sims": "Tipping Share",
                "simulated_dem_win_prob": "Dem Win Prob",
                "model_margin_dem": "Model Margin",
            }), use_container_width=True, hide_index=True)

with tab_races:
    st.subheader("Race-Level Forecast")
    if race_stats.empty:
        st.info("No race stats found.")
    else:
        display = race_stats.copy()
        display["Dem Win Prob"] = display["simulated_dem_win_prob"].map(lambda v: f"{v:.1%}")
        display["Pre-Sim Prob"] = display["pre_sim_dem_win_prob"].map(lambda v: f"{v:.1%}")
        display["Model Margin"] = display["model_margin_dem"].map(lambda v: f"{v:.1f}")
        display["Avg Sim Margin"] = display["avg_simulated_margin_dem"].map(lambda v: f"{v:.1f}")
        cols = [
            "state", "race_tier", "dem_candidate", "gop_candidate", "current_holder",
            "Model Margin", "Dem Win Prob", "Avg Sim Margin",
            "elasticity", "dem_win_counts_for_seat_change"
        ]
        st.dataframe(display[cols].rename(columns={
            "state": "State",
            "race_tier": "Tier",
            "dem_candidate": "Dem Candidate",
            "gop_candidate": "GOP Candidate",
            "current_holder": "Current Holder",
            "elasticity": "Elasticity",
            "dem_win_counts_for_seat_change": "Seat Gain If Dem Wins",
        }), use_container_width=True, hide_index=True)

        chart_df = race_stats.sort_values("simulated_dem_win_prob").copy()
        chart_df["Favored Party"] = chart_df["simulated_dem_win_prob"].apply(
            lambda p: "Democrat" if float(p) >= 0.5 else "Republican"
        )

        fig = px.bar(
            chart_df,
            x="simulated_dem_win_prob",
            y="state",
            orientation="h",
            color="Favored Party",
            color_discrete_map={
                "Democrat": DEM_COLOR,
                "Republican": GOP_COLOR,
            },
            labels={"simulated_dem_win_prob": "Democratic Win Probability", "state": "State"},
        )
        fig.update_layout(xaxis_tickformat=".0%", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with tab_scenarios:
    st.subheader("National Swing Sensitivity")
    if scenarios.empty:
        st.info("No scenario outputs found. Run `python scenario_runner.py`.")
    else:
        fig = px.line(
            scenarios,
            x="national_environment_margin_dem",
            y="dem_control_probability",
            markers=True,
            labels={
                "national_environment_margin_dem": "National Swing Toward Democrats",
                "dem_control_probability": "Democratic Control Probability",
            },
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(scenarios, use_container_width=True, hide_index=True)

with tab_polls:
    st.subheader("Weighted Polling Averages")
    if poll_avgs.empty:
        st.info("No generated polling averages found.")
    else:
        st.dataframe(poll_avgs, use_container_width=True, hide_index=True)

    st.subheader("Bayesian Update Audit")
    if bayes.empty:
        st.info("No Bayesian update file found.")
    else:
        st.dataframe(bayes, use_container_width=True, hide_index=True)

with tab_method:
    st.subheader("Methodology")
    st.markdown("""
    **Pipeline**

    1. Enter/export polls into `inputs/polls_raw.csv`.
    2. `ingest_polls.py` creates weighted polling averages.
    3. `bayesian_update.py` combines fundamentals and polling.
    4. `run_model.py` simulates race margins using shared national and race-specific errors.
    5. `scenario_runner.py` tests national swing sensitivity.
    6. This dashboard reads the output CSVs.

    **Key modeling choices**

    - Outcomes are simulated from margins.
    - Shared national error creates correlated outcomes.
    - Race-specific error captures local uncertainty.
    - Time-to-election calibration controls uncertainty and polling weight.
    - Candidate metadata is separate from numerical assumptions.
    """)

    st.code("""
python run_full_pipeline.py --as-of 2026-05-22 --days-out 165 --sims 50000
streamlit run dashboard_app.py
    """.strip(), language="bash")


# Manual poll entry form
st.subheader("Add Manual Poll")

manual_raw_path = "inputs/manual_polls.csv"

with st.form("manual_poll_entry_form"):
    race = st.text_input("Race", value="ME Senate")
    state = st.text_input("State abbreviation", value="ME")
    chamber = st.selectbox("Chamber", ["Senate", "House", "Governor"], index=0)
    pollster = st.text_input("Pollster")
    
    pollster_grade = st.selectbox(
        "Pollster grade",
        ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "Unknown"],
        index=10
    )
    
    house_effect_dem = st.number_input(
    "House effect, Dem points",
    min_value=-10.0,
    max_value=10.0,
    value=0.0,
    step=0.1,
    help="Placeholder for now. Positive means the pollster tends to favor Democrats; negative means it tends to favor Republicans."
)


    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")
    
    sample_size = st.number_input("Sample size", min_value=1, value=800, step=1)
    sample_type = st.selectbox("Sample type", ["LV", "RV", "A", "Other"], index=0)

    dem_candidate = st.text_input("Democratic candidate")
    rep_candidate = st.text_input("Republican candidate")
    ind_candidate = st.text_input("Independent candidate")
    other_candidate = st.text_input("Other candidate")

    dem_pct = st.number_input("Dem %", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
    rep_pct = st.number_input("Rep %", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
    ind_pct = st.number_input("Independent %", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    other_pct = st.number_input("Other %", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    undecided_pct = st.number_input("Undecided %", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    notes = st.text_area("Notes")

    submitted = st.form_submit_button("Add poll")

    if submitted:

        new_poll = pd.DataFrame([{
            "race": race,
            "state": inferred_state if inferred_state else state,
            "chamber": chamber,
            "pollster": pollster,
            "pollster_grade": pollster_grade,
            "house_effect_dem": house_effect_dem,
            "start_date": start_date,
            "end_date": end_date,
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
            "notes": notes
        }])

        if Path(manual_raw_path).exists():
            existing_polls = pd.read_csv(manual_raw_path)
            updated_polls = pd.concat([existing_polls, new_poll], ignore_index=True)
        else:
            updated_polls = new_poll

        updated_polls.to_csv(manual_raw_path, index=False)

        subprocess.run(["python3", "validate_manual_polls.py"])

        subprocess.run(["python3", "run_full_pipeline.py"])

        st.success("Poll added, validated, and full model pipeline refreshed successfully.")

        if Path(manual_raw_path).exists():
            existing_polls = pd.read_csv(manual_raw_path)
            updated_polls = pd.concat([existing_polls, new_poll], ignore_index=True)
        else:
            updated_polls = new_poll

        updated_polls.to_csv(manual_raw_path, index=False)

        subprocess.run(["python3", "validate_manual_polls.py"])

        subprocess.run(["python3", "run_full_pipeline.py"])

        st.success("Poll added, validated, and full model pipeline refreshed successfully.")
# Editable manual poll management table
st.subheader("Manage Manual Polls")

manual_raw_path = "inputs/manual_polls.csv"

if Path(manual_raw_path).exists():
    manual_polls_editor = pd.read_csv(manual_raw_path)

    if not manual_polls_editor.empty:
        st.caption("Edit cells directly. To delete a poll, select the row using the left checkbox column and click Save Changes.")

        edited_manual_polls = st.data_editor(
            manual_polls_editor,
            use_container_width=True,
            num_rows="dynamic",
            key="manual_polls_editor"
        )

        if st.button("Save Manual Poll Changes"):
            edited_manual_polls.to_csv(manual_raw_path, index=False)

            subprocess.run(["python3", "validate_manual_polls.py"])
            subprocess.run(["python3", "run_full_pipeline.py"])

            st.success("Manual poll changes saved, validated, and model refreshed.")
    else:
        st.info("No manual polls available to edit yet.")
else:
    st.info("No manual_polls.csv file found yet.")

# Recent manual polls table
st.subheader("Recent Manual Polls")

manual_polls_path = "outputs/manual_polls_clean.csv"

if Path(manual_polls_path).exists():
    manual_polls_display = pd.read_csv(manual_polls_path)

    if not manual_polls_display.empty:
        manual_polls_display["end_date"] = pd.to_datetime(
            manual_polls_display["end_date"],
            errors="coerce"
        )

        recent_manual_polls = (
            manual_polls_display
            .sort_values("end_date", ascending=False)
            .head(30)
        )

        display_cols = [
            "race",
            "pollster",
            "end_date",
            "sample_size",
            "sample_type",
            "dem_pct",
            "rep_pct",
            "ind_pct",
            "leader",
            "leader_margin",
            "poll_weight",
            "is_three_way_race",
            "notes"
        ]

        display_cols = [
            col for col in display_cols
            if col in recent_manual_polls.columns
        ]

        st.dataframe(
            recent_manual_polls[display_cols],
            use_container_width=True
        )
    else:
        st.info("No manual polls found yet.")
else:
    st.info("No manual poll file found yet. Run validate_manual_polls.py first.")
    # Model Inputs Audit Panel
st.subheader("Model Inputs Audit")

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}


def normalize_state_for_audit(x):
    x = str(x).strip().upper()
    return STATE_NAME_TO_ABBR.get(x, x)


race_inputs_path = INPUTS / "race_inputs.csv"
polling_avgs_path = INPUTS / "polling_averages_generated.csv"
bayesian_path = INPUTS / "bayesian_update_generated.csv"

if race_inputs_path.exists():
    race_inputs_audit = pd.read_csv(race_inputs_path)
else:
    race_inputs_audit = pd.DataFrame()

if polling_avgs_path.exists():
    polling_audit = pd.read_csv(polling_avgs_path)
else:
    polling_audit = pd.DataFrame()

if bayesian_path.exists():
    bayesian_audit = pd.read_csv(bayesian_path)
else:
    bayesian_audit = pd.DataFrame()

if race_inputs_audit.empty:
    st.info("No race_inputs.csv file found for audit.")
else:
    audit_df = race_inputs_audit.copy()

    # Normalize state identifiers across all audit inputs
    if "state" in audit_df.columns:
        audit_df["state"] = audit_df["state"].apply(normalize_state_for_audit)

    if not polling_audit.empty and "state" in polling_audit.columns:
        polling_audit["state"] = polling_audit["state"].apply(normalize_state_for_audit)

    if not bayesian_audit.empty and "state" in bayesian_audit.columns:
        bayesian_audit["state"] = bayesian_audit["state"].apply(normalize_state_for_audit)

    # Merge manual-only polling averages
    if not polling_audit.empty and "state" in polling_audit.columns:
        poll_cols = [
            col for col in [
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
            if col in polling_audit.columns
        ]

        polling_for_merge = polling_audit[poll_cols].copy()

        polling_for_merge = polling_for_merge.rename(
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

        audit_df = audit_df.merge(
            polling_for_merge,
            on="state",
            how="left"
        )

    # Merge Bayesian/final blended model values
    if not bayesian_audit.empty and "state" in bayesian_audit.columns:
        bayes_cols = [
            col for col in [
                "state",
                "prior_margin_dem",
                "fundamentals_margin_dem",
                "polling_margin_dem",
                "polling_margin_used",
                "posterior_margin_dem",
                "posterior_sd",
                "polling_sd",
                "bayesian_prior_weight",
                "bayesian_polling_weight",
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
                "poll_weight",
                "fundamentals_weight",
                "polling_weight",
                "final_margin_dem",
                "model_margin_dem",
            ]
            if col in bayesian_audit.columns
        ]

        bayes_for_merge = bayesian_audit[bayes_cols].copy()

        if "polling_margin_dem" in bayes_for_merge.columns:
            bayes_for_merge = bayes_for_merge.rename(
                columns={"polling_margin_dem": "bayesian_polling_margin_dem"}
            )

        audit_df = audit_df.merge(
            bayes_for_merge,
            on="state",
            how="left"
        )

    # Polling status
    if "manual_poll_count" in audit_df.columns:
        audit_df["manual_poll_count"] = pd.to_numeric(
            audit_df["manual_poll_count"],
            errors="coerce"
        ).fillna(0)

        audit_df["polling_status"] = audit_df["manual_poll_count"].apply(
            lambda x: "Manual polling used" if x > 0 else "Fundamentals only"
        )
    else:
        audit_df["manual_poll_count"] = 0
        audit_df["polling_status"] = "Fundamentals only"

    # Polling effect diagnostic
    if "manual_polling_margin_dem" in audit_df.columns:
        audit_df["manual_polling_margin_dem"] = pd.to_numeric(
            audit_df["manual_polling_margin_dem"],
            errors="coerce"
        )

    if "fundamentals_margin_dem" in audit_df.columns:
        audit_df["fundamentals_margin_dem"] = pd.to_numeric(
            audit_df["fundamentals_margin_dem"],
            errors="coerce"
        )

    if (
        "manual_polling_margin_dem" in audit_df.columns
        and "fundamentals_margin_dem" in audit_df.columns
    ):
        audit_df["poll_vs_fundamentals_gap"] = (
            audit_df["manual_polling_margin_dem"]
            - audit_df["fundamentals_margin_dem"]
        )

    if (
        "posterior_margin_dem" in audit_df.columns
        and "fundamentals_margin_dem" in audit_df.columns
    ):
        audit_df["model_movement_from_fundamentals"] = (
            audit_df["posterior_margin_dem"]
            - audit_df["fundamentals_margin_dem"]
        )

    # RCV/election-system defaults
    if "election_system" not in audit_df.columns:
        audit_df["election_system"] = "plurality"

    if "rcv_enabled" not in audit_df.columns:
        audit_df["rcv_enabled"] = False

    audit_df["rcv_enabled"] = audit_df["rcv_enabled"].astype(str)

    # Choose display columns that exist
    preferred_cols = [
        "state",
        "race",
        "dem_candidate",
        "rep_candidate",
        "ind_candidate",
        "incumbent_party",
        "incumbent_running",
        "open_seat",
        "election_system",
        "rcv_enabled",
        "polling_status",
        "fundamentals_margin_dem",
        "manual_polling_margin_dem",
        "bayesian_polling_margin_dem",
        "posterior_margin_dem",
        "final_margin_dem",
        "model_margin_dem",
        "posterior_sd",
        "poll_vs_fundamentals_gap",
        "model_movement_from_fundamentals",
        "manual_poll_count",
        "manual_effective_poll_count",
        "manual_latest_poll_end_date",
        "manual_avg_poll_age_days",
        "manual_total_poll_weight",
        "manual_largest_pollster_weight_share",
        "manual_only_partisan_or_internal_polls",
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
        "bayesian_polling_weight",
        "bayesian_prior_weight",
        "polling_sd",
        "poll_weight",
        "polling_weight",
        "fundamentals_weight",
    ]

    display_cols = [
        col for col in preferred_cols
        if col in audit_df.columns
    ]

    st.dataframe(
        audit_df[display_cols],
        use_container_width=True
    )

    # Summary check
    manual_states = audit_df.loc[
        audit_df["polling_status"] == "Manual polling used",
        "state"
    ].tolist()

    if manual_states:
        st.caption(
            "Manual polling currently detected for: "
            + ", ".join(sorted(manual_states))
        )
    else:
        st.caption("No manual polling detected in the audit table.") 

# -----------------------------
# Fundamentals Audit Panel
# -----------------------------
st.divider()
st.subheader("Fundamentals Audit")

st.caption(
    "Breaks down how each race's fundamentals are built: presidential baseline, "
    "national environment through elasticity, race-specific adjustments, polling, "
    "Bayesian model margin, and simulated win probability."
)

fundamentals_race_path = INPUTS / "race_inputs.csv"
fundamentals_polling_path = INPUTS / "polling_averages_generated.csv"
fundamentals_stats_path = OUTPUTS / "race_stats.csv"
fundamentals_env_path = INPUTS / "national_environment.csv"

if fundamentals_race_path.exists():
    fundamentals_df = pd.read_csv(fundamentals_race_path)
else:
    fundamentals_df = pd.DataFrame()

if fundamentals_polling_path.exists():
    fundamentals_polling = pd.read_csv(fundamentals_polling_path)
else:
    fundamentals_polling = pd.DataFrame()

if fundamentals_stats_path.exists():
    fundamentals_stats = pd.read_csv(fundamentals_stats_path)
else:
    fundamentals_stats = pd.DataFrame()

if fundamentals_env_path.exists():
    fundamentals_env = pd.read_csv(fundamentals_env_path)
else:
    fundamentals_env = pd.DataFrame()


def normalize_state_code(x):
    return str(x).strip().upper()


if fundamentals_df.empty:
    st.info("No race_inputs.csv file found for fundamentals audit.")
else:
    audit = fundamentals_df.copy()

    if "state" in audit.columns:
        audit["state"] = audit["state"].apply(normalize_state_code)

    if not fundamentals_polling.empty and "state" in fundamentals_polling.columns:
        fundamentals_polling["state"] = fundamentals_polling["state"].apply(normalize_state_code)

        polling_cols = [
            col for col in [
                "state",
                "polling_margin_dem",
                "poll_count",
                "latest_poll_end_date",
                "avg_poll_age_days",
                "total_poll_weight",
            ]
            if col in fundamentals_polling.columns
        ]

        polling_for_audit = fundamentals_polling[polling_cols].copy()

        polling_for_audit = polling_for_audit.rename(
            columns={
                "polling_margin_dem": "manual_polling_margin_dem",
                "poll_count": "manual_poll_count",
                "latest_poll_end_date": "manual_latest_poll_end_date",
                "avg_poll_age_days": "manual_avg_poll_age_days",
                "total_poll_weight": "manual_total_poll_weight",
            }
        )

        audit = audit.merge(
            polling_for_audit,
            on="state",
            how="left"
        )

    if not fundamentals_stats.empty and "state" in fundamentals_stats.columns:
        fundamentals_stats["state"] = fundamentals_stats["state"].apply(normalize_state_code)

        stats_cols = [
            col for col in [
                "state",
                "model_margin_dem",
                "simulated_dem_win_prob",
                "avg_simulated_margin_dem",
                "pre_sim_dem_win_prob",
            ]
            if col in fundamentals_stats.columns
        ]

        stats_for_audit = fundamentals_stats[stats_cols].copy()

        stats_for_audit = stats_for_audit.rename(
            columns={
                "model_margin_dem": "final_model_margin_dem",
                "avg_simulated_margin_dem": "avg_simulated_margin_dem",
                "simulated_dem_win_prob": "simulated_dem_win_prob",
                "pre_sim_dem_win_prob": "pre_sim_dem_win_prob",
            }
        )

        audit = audit.merge(
            stats_for_audit,
            on="state",
            how="left"
        )

    # Numeric cleanup
    numeric_cols = [
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_partisan_baseline_dem",
        "state_elasticity",
        "national_environment_margin_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_margin_dem",
        "manual_polling_margin_dem",
        "manual_poll_count",
        "bayesian_model_margin_dem",
        "bayesian_polling_weight",
        "bayesian_posterior_sd",
        "final_model_margin_dem",
        "avg_simulated_margin_dem",
        "pre_sim_dem_win_prob",
        "simulated_dem_win_prob",
    ]

    for col in numeric_cols:
        if col in audit.columns:
            audit[col] = pd.to_numeric(
                audit[col],
                errors="coerce"
            )

    # Derived explanatory columns
    if (
        "fundamentals_margin_dem" in audit.columns
        and "state_partisan_baseline_dem" in audit.columns
    ):
        audit["fundamentals_vs_baseline_shift"] = (
            audit["fundamentals_margin_dem"]
            - audit["state_partisan_baseline_dem"]
        )

    if (
        "bayesian_model_margin_dem" in audit.columns
        and "fundamentals_margin_dem" in audit.columns
    ):
        audit["bayesian_shift_from_fundamentals"] = (
            audit["bayesian_model_margin_dem"]
            - audit["fundamentals_margin_dem"]
        )

    if (
        "final_model_margin_dem" in audit.columns
        and "fundamentals_margin_dem" in audit.columns
    ):
        audit["final_shift_from_fundamentals"] = (
            audit["final_model_margin_dem"]
            - audit["fundamentals_margin_dem"]
        )

    # National environment summary
    if not fundamentals_env.empty:
        env_latest = fundamentals_env.iloc[-1]

        env_cols = [
            "as_of_date",
            "generic_ballot_margin_dem",
            "presidential_approval",
            "presidential_disapproval",
            "presidential_net_approval",
            "approval_adjustment_dem",
            "midterm_adjustment_dem",
            "national_environment_margin_dem",
            "source_notes",
        ]

        env_display = {
            col: env_latest[col]
            for col in env_cols
            if col in fundamentals_env.columns
        }

        with st.expander("National Environment Inputs", expanded=True):
            st.dataframe(
                pd.DataFrame([env_display]),
                use_container_width=True
            )

    preferred_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "current_holder",
        "race_tier",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_partisan_baseline_dem",
        "national_environment_margin_dem",
        "state_elasticity",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_margin_dem",
        "fundamentals_vs_baseline_shift",
        "manual_polling_margin_dem",
        "manual_poll_count",
        "bayesian_model_margin_dem",
        "bayesian_polling_weight",
        "bayesian_posterior_sd",
        "bayesian_shift_from_fundamentals",
        "final_model_margin_dem",
        "final_shift_from_fundamentals",
        "simulated_dem_win_prob",
        "fundamentals_notes",
    ]

    display_cols = [
        col for col in preferred_cols
        if col in audit.columns
    ]

    st.dataframe(
        audit[display_cols],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Positive margins favor Democrats; negative margins favor Republicans. "
        "The national environment adjustment equals national environment margin × state elasticity."
    )