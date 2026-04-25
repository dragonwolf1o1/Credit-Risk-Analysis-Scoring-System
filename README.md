# Credit Risk Analysis and Scoring System

A professional end-to-end credit risk analysis project that combines MySQL-based data modeling, feature engineering, machine learning, automated loan scoring, and export-ready reporting.

This repository is designed to simulate a practical banking and lending analytics workflow: customer and loan data is stored in MySQL, transformed into a model-ready dataset, validated with quality guardrails, scored for default probability, and exported for business reporting.

![Data Pipeline Architecture](docs/Data%20Pipeline%20Architecture.png)

## Overview

Financial institutions need fast and reliable ways to identify risky borrowers, reduce manual review effort, and support better lending decisions. This project addresses that problem by building a structured credit risk scoring pipeline with:

- customer, loan, and repayment data stored in MySQL
- a feature engineering layer for borrower and repayment behavior
- a logistic regression model for default probability prediction
- calibrated risk-band assignment for decision support
- automated scoring for newly added loans
- validation checks that reject bad records before they pollute training or scoring

## Key Capabilities

- End-to-end training and scoring workflow using Python, MySQL, pandas, and scikit-learn
- Centralized feature engineering shared across training and scoring
- Input validation rules to reduce data-entry and transformation errors
- Threshold tuning and risk-band calibration based on validation output
- Rejection logging for invalid records during scoring
- CSV and Excel export generation for reporting and downstream analysis
- Backward-compatible root scripts for easier execution

## Project Architecture

The solution is organized into a small reusable package plus script entrypoints:

```text
Credit Risk Analysis and Scoring System/
├── credit_risk/              # Core application logic
│   ├── config.py             # Settings, paths, environment configuration
│   ├── constants.py          # Shared field lists and allowed values
│   ├── db.py                 # MySQL engine and rejection logging helpers
│   ├── features.py           # Shared feature engineering
│   ├── scoring.py            # Scoring workflow and daemon logic
│   ├── thresholding.py       # Decision threshold and risk-band calibration
│   ├── training.py           # Training pipeline and reporting
│   └── validation.py         # Data quality and rule-based validation
├── scripts/                  # Clean executable entrypoints
├── sql/                      # Database schema and view definitions
├── artifacts/
│   ├── models/               # Saved trained model bundles
│   └── reports/              # Training reports and rejected row files
├── data/
│   └── exports/              # CSV and Excel outputs
├── docs/                     # Project diagrams and documentation assets
└── *.py                      # Legacy-friendly wrapper scripts
```

## Workflow

### 1. Data Layer

The MySQL schema contains:

- `customers`
- `loan`
- `repayments`
- `model_score`
- `model_score_rejections`

The project uses the `vm_model_dataset` view to create one model-ready row per loan by joining borrower details, loan attributes, and repayment aggregates.

### 2. Feature Engineering

The shared feature pipeline derives business-relevant variables such as:

- `age_years`
- `loan_to_income_ratio`
- `missed_payment_percentage`
- `is_secured`

This logic is centralized so training and scoring stay consistent.

### 3. Model Training

The training flow:

- loads data from `vm_model_dataset`
- applies feature engineering
- validates row quality
- rejects invalid records and saves them to a report
- performs stratified validation when the dataset allows
- tunes the default-decision threshold
- calibrates risk-band boundaries
- saves a model bundle and a training report

### 4. Loan Scoring

The scoring flow:

- fetches loans that have not yet been scored
- applies the same feature engineering rules
- validates each row before scoring
- logs invalid loans into `model_score_rejections`
- generates default probabilities and risk bands
- writes valid scores to `model_score`

## Data Quality and Model Governance

One of the main strengths of the current version is the addition of quality guardrails.

The system now checks for issues such as:

- missing or invalid `loan_id`, `customer_id`, or `date_of_birth`
- negative or zero income and loan amounts
- invalid interest rate and loan term ranges
- inconsistent repayment aggregates
- unsupported categorical values
- duplicate loan identifiers in the model dataset

This reduces scoring errors, improves trust in model outputs, and makes the pipeline more production-oriented.

## Technology Stack

- Python
- MySQL
- pandas
- SQLAlchemy
- scikit-learn
- joblib
- PowerShell

## Setup

### Prerequisites

- MySQL installed and running locally
- Python environment available in `risk_analysis_env`
- Required Python packages installed in that environment

### Database Setup

Run the schema file:

```sql
SOURCE sql/schema.sql;
```

This creates the database objects required for training and scoring.

### Environment Variables

You can override default configuration values with:

- `CRS_DB_USER`
- `CRS_DB_PASSWORD`
- `CRS_DB_HOST`
- `CRS_DB_NAME`
- `CRS_MODEL_VERSION`
- `CRS_POLL_SECONDS`

## How To Run

### Check Database Connectivity

```powershell
.\risk_analysis_env\Scripts\python.exe "mysql_connection_&_fetch_data.py"
```

### Inspect the Training Dataset

```powershell
.\risk_analysis_env\Scripts\python.exe dataset_for_model.py
```

### Train the Model

```powershell
.\risk_analysis_env\Scripts\python.exe train_model.py
```

### Score Pending Loans Once

```powershell
.\risk_analysis_env\Scripts\python.exe score_loan_once.py
```

### Run Continuous Auto-Scoring

```powershell
.\risk_analysis_env\Scripts\python.exe Score_Loan_Deamon.py
```

### Export Scored Results

```powershell
.\risk_analysis_env\Scripts\python.exe extract_data_for_excel.py
```

## Outputs

After execution, the project can generate:

- trained model bundle in `artifacts/models/`
- training report in `artifacts/reports/`
- rejected training rows in `artifacts/reports/`
- rejected scoring rows in `artifacts/reports/`
- scored loan exports in `data/exports/`

## Main Files

- `credit_risk/training.py` handles model training, evaluation, threshold tuning, and artifact generation
- `credit_risk/scoring.py` handles one-time and daemon-based scoring workflows
- `credit_risk/validation.py` contains record-level validation rules
- `credit_risk/features.py` contains reusable feature engineering logic
- `sql/schema.sql` contains the organized database schema and model-supporting tables

## Use Cases

This project can be presented as:

- a finance and banking analytics portfolio project
- a machine learning pipeline case study
- a credit scoring proof of concept
- a demonstration of data quality controls in risk systems

## Future Enhancements

- add model comparison across multiple algorithms
- build a dashboard for score monitoring and portfolio trends
- add unit tests for validation and feature engineering rules
- track model drift and retraining triggers
- support API-based real-time scoring

## Author Notes

This repository is structured to show both analytical thinking and software organization. It is suitable for demonstrating practical skills in:

- risk analytics
- machine learning implementation
- SQL data modeling
- data validation
- automation and reporting

---

If you want, I can also make the README look even more polished with:

- a stronger portfolio-style project summary
- resume-friendly highlights
- business impact wording
- installation badges and a cleaner visual header
