# Q1: Automatic Calendar + Candidate Refresh

## Main change

The model no longer requires hard-coded `--as-of` or `--days-out` arguments in normal use.

Recommended commands:

```bash
python3 run_full_pipeline.py --sims 50000
streamlit run dashboard_app.py
```

Or, with aggregator import:

```bash
python3 run_full_pipeline.py --import-aggregators --presidential-approval 38.4 --sims 50000
streamlit run dashboard_app.py
```

## VS Code tasks

The package now includes `.vscode/tasks.json` with:

- Run Senate Model Pipeline
- Run Pipeline with Aggregators
- Launch Streamlit Dashboard

## Candidate/status refresh

`inputs/race_inputs.csv` was refreshed as of 2026-05-23 for retirements, open seats, appointees, primary outcomes, and unresolved runoffs/primaries.

Candidate status changes quickly during primary season; the `candidate_status_as_of` column is included for auditability.