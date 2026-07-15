# 2026 Senate Forecast Model — M1 Python Migration

This is the first Python-native version of the Senate model.

## What it does

- Reads race inputs from CSV.
- Blends polling and fundamentals based on days until Election Day.
- Applies a national environment adjustment using generic ballot, presidential approval, midterm effect, and a manual scenario input.
- Simulates outcomes from **margins**, not from fixed win probabilities.
- Uses shared national error plus race-specific error to produce correlated election outcomes.
- Writes forecast summary, race stats, seat distribution, and simulation draws to CSV.

## Quick start

```bash
pip install -r requirements.txt
python run_model.py
```

Useful variants:

```bash
python run_model.py --sims 50000 --today 2026-08-01
python scenario_runner.py
```

## Inputs

### `inputs/race_inputs.csv`

Important columns:

- `polling_margin_dem`
- `fundamentals_margin_dem`
- `elasticity`
- `dem_win_counts_for_seat_change`

`dem_win_counts_for_seat_change = 1` means a Democratic win adds a seat relative to the baseline.  
`0` means that race is already included in the baseline.

### `inputs/national_environment.csv`

Controls generic ballot, presidential approval, midterm effect, and manual national swing.

### `inputs/calibration_parameters.csv`

Controls time-varying uncertainty, correlation, and polling weight.

## Outputs

Written to `outputs/`:

- `forecast_summary.csv`
- `forecast_summary.json`
- `race_stats.csv`
- `seat_distribution.csv`
- `simulation_draws.csv`
- `scenario_summary.csv` after running `scenario_runner.py`

## Modeling note

This is deliberately transparent and editable. It is meant to replace the fragile spreadsheet simulation engine while preserving the core concepts we developed: national environment, dynamic polling weight, time-varying uncertainty, and correlated errors.


## Candidate data freshness

Candidate metadata in `inputs/race_inputs.csv` was refreshed on 2026-05-22. See `CANDIDATE_REFRESH_NOTES.md`.


## M2 poll ingestion

This package adds `ingest_polls.py` and `senate_model/poll_ingestion.py`. See `M2_POLL_INGESTION_README.md`.


## M3 Bayesian updater

This package adds `bayesian_update.py` and `senate_model/bayesian_updater.py`. See `M3_BAYESIAN_UPDATER_README.md`.


## N dashboard

This package adds `dashboard_app.py` and `run_full_pipeline.py`. See `N_DASHBOARD_README.md`.


## O1 full race universe

Expanded to all 35 2026 Senate elections. Baseline seats now default to 34. See `O1_FULL_RACE_UNIVERSE_README.md`.


## P1 aggregator ingestion

This package adds `import_aggregators.py` and an `inputs/aggregator_sources.csv` configuration file. See `P1_AGGREGATOR_INGESTION_README.md`.


## Q1 automatic calendar + candidate refresh

See `Q1_AUTO_CALENDAR_CANDIDATE_REFRESH_README.md`.
