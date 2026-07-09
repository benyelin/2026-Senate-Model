import argparse
from datetime import date
import subprocess
import sys


def today_iso():
    return date.today().isoformat()


def compute_days_out():
    election_day = date(2026, 11, 3)
    return max(0, (election_day - date.today()).days)


def run(cmd):
    print("\n$", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full Senate model pipeline.")
    parser.add_argument("--as-of", default=None, help="Optional YYYY-MM-DD as-of date. If omitted, uses today.")
    parser.add_argument("--days-out", type=int, default=None, help="Optional days until Election Day. If omitted, computed automatically.")
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--import-aggregators", action="store_true", help="Import configured aggregator data before ingesting polls.")
    parser.add_argument("--presidential-approval", type=float, default=None, help="Optional approval value to write into national environment.")
    args = parser.parse_args()

    as_of = args.as_of or today_iso()
    days_out = args.days_out if args.days_out is not None else compute_days_out()

    py = sys.executable
    if args.import_aggregators:
        cmd = [py, "import_aggregators.py", "--national", "--races"]
        if args.presidential_approval is not None:
            cmd.extend(["--presidential-approval", str(args.presidential_approval)])
        run(cmd)
    run([py, "update_candidate_quality.py"])
    run([py, "sync_candidate_quality_fields.py"])
    run([py, "recalculate_fundamentals.py"])
    run([py, "build_senate_candidate_war_audit.py"])
    run([py, "build_senate_candidate_quality_framework_audit.py"])
    run([py, "apply_senate_candidate_quality_framework.py", "--apply"])
    run([py, "audit_senate_fundamentals_components.py"])
    run([py, "apply_senate_partisan_pollster_adjustments.py"])
    run([py, "validate_manual_polls.py"])
    run([py, "ingest_polls.py", "--as-of", as_of])
    run([py, "apply_candidate_status_overrides.py"])
    run([py, "bayesian_update.py", "--days-out", str(days_out)])
    run([py, "apply_senate_bayesian_weight_cap.py", "--days-out", str(days_out)])
    run([py, "sync_bayesian_poll_metadata.py"])
    run([py, "cap_bayesian_poll_weight.py"])
    run([py, "sync_senate_model_fields.py"])
    run([py, "run_model.py", "--today", as_of, "--sims", str(args.sims)])
    run([py, "append_senate_forecast_history.py"])
    run([py, "scenario_runner.py"])

    print("\nPipeline complete.")
    print("Launch dashboard:")
    print("  streamlit run dashboard_app.py")


if __name__ == "__main__":
    main()
