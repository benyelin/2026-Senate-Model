# P1: Aggregator Ingestion

This version adds a semi-automated importer for public polling aggregator data.

## New files

- `import_aggregators.py`
- `senate_model/aggregator_ingestion.py`
- `inputs/aggregator_sources.csv`

## What it can update

### National environment

It can try to pull the generic ballot from enabled sources in `inputs/aggregator_sources.csv`.

It can also write presidential approval if you pass it manually:

```bash
python3 import_aggregators.py --national --presidential-approval 38.4
```

### Race polling

It can try to parse enabled Senate race pages and add synthetic aggregator-average rows to `inputs/polls_raw.csv`.

This is intentionally conservative because aggregator pages differ by race and matchup.

## Recommended workflow

```bash
python3 import_aggregators.py --national --races --presidential-approval 38.4
python3 run_full_pipeline.py --as-of 2026-05-22 --days-out 165 --sims 50000
streamlit run dashboard_app.py
```

Or combined:

```bash
python3 run_full_pipeline.py --import-aggregators --presidential-approval 38.4 --as-of 2026-05-22 --days-out 165 --sims 50000
streamlit run dashboard_app.py
```

## Important caveat

Sites like RCP, RealClearPolling, 270toWin, and Silver Bulletin can change page layouts, block scripts, or require subscription access. This importer saves audit files and raw HTML snapshots so failed parses can be checked manually.

Generated files:

- `inputs/national_environment_import_audit.csv`
- `inputs/aggregator_race_polling_generated.csv`
- `inputs/aggregator_snapshots/*.html`