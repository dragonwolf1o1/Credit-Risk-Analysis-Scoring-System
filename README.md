# Credit Risk Analysis and Scoring System

This project is now organized around a shared Python package so training, validation, and scoring use the same rules.

## Structure

- `credit_risk/`: shared config, feature engineering, validation, threshold tuning, training, and scoring logic
- `scripts/`: clean entrypoints for training, one-time scoring, daemon scoring, exports, DB checks, and dataset inspection
- `artifacts/models/`: saved model bundle
- `artifacts/reports/`: training reports and rejected-row reports
- `data/exports/`: scored loan exports in CSV and Excel
- `sql/schema.sql`: schema and view definition for MySQL

## Main commands

- `train_model.py`: train the model with validation, tuned thresholds, and saved reports
- `score_loan_once.py`: score only the current unscored loans
- `Score_Loan_Deamon.py`: run the continuous scoring loop
- `dataset_for_model.py`: inspect the dataset after feature engineering and validation
- `extract_data_for_excel.py`: export scored loans to CSV and Excel
- `mysql_connection_&_fetch_data.py`: verify DB connectivity and key table counts

## What changed

- Added row-level data validation before training and scoring
- Removed duplicated feature engineering logic from multiple scripts
- Added stratified model validation instead of evaluating only on training data
- Tuned decision threshold and risk-band thresholds from validation results
- Added rejection logging for invalid scoring rows
- Saved training metrics and rejected rows into `artifacts/reports/`

## Configuration

The project still defaults to the current MySQL settings, but you can override them with environment variables:

- `CRS_DB_USER`
- `CRS_DB_PASSWORD`
- `CRS_DB_HOST`
- `CRS_DB_NAME`
- `CRS_MODEL_VERSION`
- `CRS_POLL_SECONDS`
