"""Shared constants for the credit risk project."""

NUMERIC_FEATURES = [
    "loan_amount",
    "intrest_rate",
    "loan_terms_month",
    "annual_income",
    "avg_days_late",
    "num_30_plus_late",
    "num_missed_payments",
    "total_sheduled_payments",
    "age_years",
    "loan_to_income_ratio",
    "missed_payment_percentage",
    "is_secured",
]

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "employment_status",
    "city",
    "state",
    "loan_type",
    "colletral_type",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BASE_REQUIRED_COLUMNS = [
    "loan_id",
    "customer_id",
    "date_of_birth",
    "gender",
    "marital_status",
    "employment_status",
    "annual_income",
    "city",
    "state",
    "loan_amount",
    "intrest_rate",
    "loan_terms_month",
    "loan_type",
    "colletral_type",
    "avg_days_late",
    "num_30_plus_late",
    "num_missed_payments",
    "total_sheduled_payments",
]

TRAINING_REQUIRED_COLUMNS = BASE_REQUIRED_COLUMNS + ["default_flag"]
SCORING_REQUIRED_COLUMNS = BASE_REQUIRED_COLUMNS

TRAINING_DATASET_VIEW = "vm_model_dataset"
SCORES_TABLE = "model_score"
REJECTIONS_TABLE = "model_score_rejections"

RISK_BANDS = ("Low", "Medium", "High", "Very High")

DEFAULT_DECISION_THRESHOLD = 0.50
DEFAULT_RISK_BAND_THRESHOLDS = {
    "low_max": 0.10,
    "medium_max": 0.25,
    "high_max": 0.50,
}

TARGET_BAND_DEFAULT_RATES = {
    "Low": 0.05,
    "Medium": 0.15,
    "High": 0.35,
    "Very High": 0.60,
}

DEFAULT_MODEL_VERSION = "v2_logreg_guarded"

VALID_GENDERS = {"Male", "Female", "Other"}
VALID_MARITAL_STATUSES = {"Single", "Married", "Divorced", "Widowed"}
VALID_EMPLOYMENT_STATUSES = {"Employed", "Business", "Retired", "Student", "Unemployed"}
VALID_LOAN_TYPES = {"Home", "Education", "Business", "Personal", "Gold", "Agriculture"}

ZERO_FILL_COLUMNS = [
    "avg_days_late",
    "num_30_plus_late",
    "num_missed_payments",
    "total_sheduled_payments",
]
