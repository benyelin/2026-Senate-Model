from pathlib import Path
from datetime import date
import shutil
import subprocess
import sys
import pandas as pd

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

NATIONAL_ENV_PATH = INPUTS / "national_environment.csv"
SCENARIO_SUMMARY_PATH = OUTPUTS / "scenario_summary.csv"

ELECTION_DAY = date(2026, 11, 3)


def compute_days_out():
    return max(0, (ELECTION_DAY - date.today()).days)


def run(cmd):
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def read_current_environment():
    if not NATIONAL_ENV_PATH.exists():
        raise FileNotFoundError(f"Could not find {NATIONAL_ENV_PATH}")

    env = pd.read_csv(NATIONAL_ENV_PATH)

    if env.empty:
        raise ValueError(f"{NATIONAL_ENV_PATH} is empty")

    if "national_environment_margin_dem" not in env.columns:
        raise ValueError(
            "national_environment.csv must include national_environment_margin_dem"
        )

    current = pd.to_numeric(
        env.iloc[-1]["national_environment_margin_dem"],
        errors="coerce"
    )

    if pd.isna(current):
        raise ValueError("national_environment_margin_dem is blank or invalid")

    return float(current)


def set_environment(value, scenario_name):
    env = pd.read_csv(NATIONAL_ENV_PATH)

    # source_notes can be entirely blank, causing pandas to infer
    # float64. Scenario runs temporarily write text into this field.
    if "source_notes" in env.columns:
        env["source_notes"] = env["source_notes"].astype("object")

    if env.empty:
        raise ValueError(f"{NATIONAL_ENV_PATH} is empty")

    env.loc[env.index[-1], "national_environment_margin_dem"] = value

    if "source_notes" in env.columns:
        env.loc[
            env.index[-1],
            "source_notes"
        ] = f"Scenario run: {scenario_name}; temporary national environment {value:+.2f}"

    env.to_csv(NATIONAL_ENV_PATH, index=False)


def read_summary(scenario_name, scenario_env, base_env):
    summary_path = OUTPUTS / "forecast_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")

    summary = pd.read_csv(summary_path)

    if summary.empty:
        raise ValueError(f"{summary_path} is empty")

    row = summary.iloc[-1].to_dict()

    out = {
        "scenario": scenario_name,
        "base_national_environment_margin_dem": base_env,
        "national_environment_margin_dem": scenario_env,
        "environment_shift_from_base": scenario_env - base_env,
    }

    for col in [
        "dem_control_probability",
        "expected_dem_seats",
        "median_dem_seats",
        "total_error_sd",
        "national_error_sd",
        "race_error_sd",
        "implied_correlation",
        "polling_weight",
        "fundamentals_weight",
        "days_out",
    ]:
        out[col] = row.get(col, None)

    return out


def run_model_chain(days_out, sims):
    py = sys.executable

    run([py, "recalculate_fundamentals.py"])
    run([py, "bayesian_update.py", "--days-out", str(days_out)])
    run([py, "sync_bayesian_poll_metadata.py"])
    run([py, "cap_bayesian_poll_weight.py"])
    run([py, "run_model.py", "--sims", str(sims)])


def run_scenarios(sims=12000):
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    base_env = read_current_environment()
    days_out = compute_days_out()

    scenarios = [
        ("Current environment", base_env),
        ("Democratic environment +2", base_env + 2.0),
        ("Democratic environment +4", base_env + 4.0),
        ("Republican environment +2", base_env - 2.0),
        ("Republican environment +4", base_env - 4.0),
    ]

    backup_path = NATIONAL_ENV_PATH.with_suffix(".csv.scenario_backup")
    shutil.copy2(NATIONAL_ENV_PATH, backup_path)

    rows = []

    try:
        for scenario_name, scenario_env in scenarios:
            print()
            print(f"=== Scenario: {scenario_name} ({scenario_env:+.2f}) ===")

            set_environment(scenario_env, scenario_name)
            run_model_chain(days_out=days_out, sims=sims)

            rows.append(
                read_summary(
                    scenario_name=scenario_name,
                    scenario_env=scenario_env,
                    base_env=base_env,
                )
            )

    finally:
        shutil.copy2(backup_path, NATIONAL_ENV_PATH)
        backup_path.unlink(missing_ok=True)

        print()
        print("Restored original national_environment.csv")

        # Rebuild normal outputs after restoring the real national environment.
        print()
        print("Rebuilding normal forecast after scenario run...")
        run_model_chain(days_out=days_out, sims=sims)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        by="national_environment_margin_dem",
        ascending=True
    )

    out.to_csv(SCENARIO_SUMMARY_PATH, index=False)

    print()
    print(f"Wrote {SCENARIO_SUMMARY_PATH}")

    return out


if __name__ == "__main__":
    df = run_scenarios()
    print()
    print(df.to_string(index=False))
