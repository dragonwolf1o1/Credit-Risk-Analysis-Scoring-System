"""Feature engineering shared by training and scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ZERO_FILL_COLUMNS


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _clean_text(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    engineered = df.copy()

    numeric_columns = [
        "loan_amount",
        "intrest_rate",
        "loan_terms_month",
        "annual_income",
        "avg_days_late",
        "num_30_plus_late",
        "num_missed_payments",
        "total_sheduled_payments",
    ]
    categorical_columns = [
        "gender",
        "marital_status",
        "employment_status",
        "city",
        "state",
        "loan_type",
        "colletral_type",
    ]

    engineered = _to_numeric(engineered, numeric_columns)
    engineered = _clean_text(engineered, categorical_columns)

    if "date_of_birth" in engineered.columns:
        engineered["date_of_birth"] = pd.to_datetime(
            engineered["date_of_birth"], errors="coerce"
        )
    else:
        engineered["date_of_birth"] = pd.NaT

    for column in ZERO_FILL_COLUMNS:
        if column in engineered.columns:
            engineered[column] = engineered[column].fillna(0)

    today = pd.Timestamp.today().normalize()
    engineered["age_years"] = (
        (today - engineered["date_of_birth"]).dt.days / 365.25
    )

    annual_income = engineered["annual_income"].fillna(0)
    engineered["loan_to_income_ratio"] = engineered["loan_amount"] / (annual_income + 1.0)

    total_payments = engineered["total_sheduled_payments"].fillna(0)
    engineered["missed_payment_percentage"] = (
        engineered["num_missed_payments"] / (total_payments + 1.0)
    )

    collateral = engineered["colletral_type"].fillna("").astype(str).str.strip().str.lower()
    unsecured_values = {"", "none", "null", "nan"}
    engineered["is_secured"] = (~collateral.isin(unsecured_values)).astype(int)

    engineered.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)

    return engineered
