# FiveThirtyEight Historical Polling Archives

## Purpose

This directory contains immutable raw polling archives recovered from the Internet Archive's Wayback Machine.

These files should never be edited in place. All cleaning, normalization, filtering, and feature construction must occur in separate processed warehouse files.

## Provenance

The original FiveThirtyEight polling endpoints became unavailable after FiveThirtyEight ceased operating. Archived copies were enumerated through the Wayback CDX API and downloaded using timestamped raw-payload replay URLs.

## Preserved archives

| Dataset | Rows | Cycles | Date range | SHA-256 |
|---|---:|---|---|---|
| `senate_polls_historical.csv` | 9,292 | 2018, 2020, 2022, 2024 | 2016-11-17 through 2024-11-04 | `09ae303a94f76dd761532fce7a356fe3b5beb1e24d4e5ba343c424c372fdfabf` |
| `house_polls_historical.csv` | 4,262 | 2018, 2019, 2020, 2021, 2022, 2023, 2024 | 2017-06-25 through 2024-11-04 | `653b575f817e06f026760df4a43790e0eca0981af23b9c54e80bd6863e69977e` |
| `generic_ballot_polls_historical.csv` | 4,832 | 2018, 2020, 2022, 2024 | 2016-11-04 through 2024-11-03 | `138a2ca649a8cec8db084269b3740f0d680a90954e18a784a5d0931e1991959a` |

## Source details

### senate_polls_historical

- Official URL: `https://projects.fivethirtyeight.com/polls-page/data/senate_polls_historical.csv`
- Wayback capture timestamp: `20250306125405`
- Wayback replay URL: `https://web.archive.org/web/20250306125405id_/https://projects.fivethirtyeight.com/polls-page/data/senate_polls_historical.csv`
- Archived file: `archives/senate_polls_historical.csv`
- SHA-256: `09ae303a94f76dd761532fce7a356fe3b5beb1e24d4e5ba343c424c372fdfabf`
- Rows: 9,292
- Columns: 52
- Poll IDs: 3012
- Question IDs: 3926

### house_polls_historical

- Official URL: `https://projects.fivethirtyeight.com/polls-page/data/house_polls_historical.csv`
- Wayback capture timestamp: `20250118200335`
- Wayback replay URL: `https://web.archive.org/web/20250118200335id_/https://projects.fivethirtyeight.com/polls-page/data/house_polls_historical.csv`
- Archived file: `archives/house_polls_historical.csv`
- SHA-256: `653b575f817e06f026760df4a43790e0eca0981af23b9c54e80bd6863e69977e`
- Rows: 4,262
- Columns: 52
- Poll IDs: 1406
- Question IDs: 1861

### generic_ballot_polls_historical

- Official URL: `https://projects.fivethirtyeight.com/polls-page/data/generic_ballot_polls_historical.csv`
- Wayback capture timestamp: `20250306091811`
- Wayback replay URL: `https://web.archive.org/web/20250306091811id_/https://projects.fivethirtyeight.com/polls-page/data/generic_ballot_polls_historical.csv`
- Archived file: `archives/generic_ballot_polls_historical.csv`
- SHA-256: `138a2ca649a8cec8db084269b3740f0d680a90954e18a784a5d0931e1991959a`
- Rows: 4,832
- Columns: 44
- Poll IDs: 3815
- Question IDs: 4832

## Data-handling rules

1. Do not modify files in `archives/`.
2. Validate source hashes before every warehouse build.
3. Preserve poll-level and question-level identifiers.
4. Apply election-date cutoffs during snapshot creation to prevent look-ahead leakage.
5. Record all cleaning and exclusions in processed-data validation reports.

## Generated files

- `archive_manifest.csv`
- `archive_manifest.json`
- `README.md`
