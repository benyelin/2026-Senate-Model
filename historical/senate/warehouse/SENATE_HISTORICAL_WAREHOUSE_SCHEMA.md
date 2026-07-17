# Senate Historical Warehouse Schema

## Purpose

The Senate historical warehouse provides validated, reusable inputs for
historical backtesting of the production Senate election model.

The warehouse follows these principles:

1. Preserve raw source data.
2. Standardize processed data separately.
3. Use one canonical race identifier across all tables.
4. Validate every cycle before inclusion in backtests.
5. Add and test one forecasting component at a time.
6. Promote components only after out-of-sample validation.

## Canonical race identifier

Each regularly scheduled Senate race uses:

    {cycle}_{state}

Examples:

    2018_AZ
    2020_ME
    2022_PA

Special elections should include an explicit suffix when multiple Senate
elections occur in the same state and cycle:

    2020_GA_REGULAR
    2020_GA_SPECIAL

## Canonical election-results fields

- race_id
- cycle
- election_date
- state
- state_name
- senate_class
- election_type
- special_election
- dem_candidate
- gop_candidate
- incumbent_name
- incumbent_party
- incumbent_running
- open_seat
- dem_votes
- gop_votes
- other_votes
- total_votes
- dem_vote_share
- gop_vote_share
- dem_two_party_share
- gop_two_party_share
- actual_margin_dem
- winner_party
- major_party_contested
- source
- source_url
- source_status
- notes

## Standard conventions

### Party labels

Use:

- D
- R
- I
- Other

Candidate ballot labels may be preserved separately, but modeling party fields
must use standardized labels.

### Vote shares

All vote-share fields are stored on a 0–100 percentage-point scale.

Example:

    dem_two_party_share = 52.4
    actual_margin_dem = 4.8

### Boolean fields

Use:

- True
- False

Do not mix strings such as Yes/No or Y/N in processed datasets.

### Missing values

Use blank CSV fields and pandas NA values.

Do not use:

- 0
- Unknown
- N/A
- None

when a value is genuinely missing.

## Initial historical cycles

The initial warehouse will cover:

- 2012
- 2014
- 2016
- 2018
- 2020
- 2022

The 2024 cycle should be added after the initial pipeline and validations are
stable.

## Validation requirements

Each cycle must verify:

- unique race_id
- valid state abbreviations
- valid Senate class
- no duplicate regularly scheduled races
- special elections clearly identified
- vote totals are nonnegative
- component vote totals do not exceed total votes
- two-party shares sum to approximately 100
- actual Democratic margin matches two-party shares
- winner party agrees with the calculated margin
- all expected Senate contests are represented
- uncontested and nonstandard races are explicitly flagged

## Backtest philosophy

Historical model development must proceed incrementally:

- Layer 0: baseline
- Layer 1: national environment
- Layer 2: incumbency
- Layer 3: polling
- Layer 4: candidate quality
- Later layers: only after validation

Each layer should be evaluated using:

- margin MAE
- margin RMSE
- margin bias
- winner accuracy
- Brier score
- log loss
- expected calibration error
- expected-seat error
- interval coverage where available

Hyperparameter selection and final evaluation must be separated. Material
production changes should use nested or otherwise genuinely out-of-sample
validation.
