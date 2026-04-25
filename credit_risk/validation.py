"""Dataset validation for training and scoring."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .constants import (
    SCORING_REQUIRED_COLUMNS,
    TRAINING_REQUIRED_COLUMNS,
    VALID_EMPLOYMENT_STATUSES,
    VALID_GENDERS,
    VALID_LOAN_TYPES,
    VALID_MARITAL_STATUSES,
)


def _append_reason(reasons: pd.Series, mask: pd.Series, message: str) -> None:
    reasons.loc[mask] = reasons.loc[mask].apply(
        lambda current: message if not current else f"{current}; {message}"
    )


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _reason_breakdown(rejected_df: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for reason_text in rejected_df.get("rejection_reasons", pd.Series(dtype="object")):
        for reason in str(reason_text).split("; "):
            cleaned = reason.strip()
            if cleaned:
                counts[cleaned] += 1
    return dict(sorted(counts.items()))


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )


def _validate_rows(df: pd.DataFrame, *, require_target: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    working = df.copy()
    reasons = pd.Series("", index=working.index, dtype="object")

    duplicate_loan_ids = working["loan_id"].duplicated(keep=False)
    _append_reason(reasons, duplicate_loan_ids, "duplicate loan_id in dataset")

    _append_reason(reasons, working["loan_id"].isna(), "missing loan_id")
    _append_reason(reasons, working["customer_id"].isna(), "missing customer_id")

    _append_reason(
        reasons,
        working["date_of_birth"].isna(),
        "invalid or missing date_of_birth",
    )
    _append_reason(
        reasons,
        working["age_years"].isna() | (working["age_years"] < 18) | (working["age_years"] > 100),
        "age_years must be between 18 and 100",
    )

    _append_reason(
        reasons,
        working["loan_amount"].isna() | (working["loan_amount"] <= 0),
        "loan_amount must be greater than 0",
    )
    _append_reason(
        reasons,
        working["annual_income"].isna() | (working["annual_income"] <= 0),
        "annual_income must be greater than 0",
    )
    _append_reason(
        reasons,
        working["intrest_rate"].isna()
        | (working["intrest_rate"] < 0)
        | (working["intrest_rate"] > 60),
        "intrest_rate must be between 0 and 60",
    )
    _append_reason(
        reasons,
        working["loan_terms_month"].isna()
        | (working["loan_terms_month"] <= 0)
        | (working["loan_terms_month"] > 480),
        "loan_terms_month must be between 1 and 480",
    )

    _append_reason(
        reasons,
        working["avg_days_late"].isna() | (working["avg_days_late"] < 0),
        "avg_days_late cannot be negative",
    )
    _append_reason(
        reasons,
        working["num_30_plus_late"].isna() | (working["num_30_plus_late"] < 0),
        "num_30_plus_late cannot be negative",
    )
    _append_reason(
        reasons,
        working["num_missed_payments"].isna() | (working["num_missed_payments"] < 0),
        "num_missed_payments cannot be negative",
    )
    _append_reason(
        reasons,
        working["total_sheduled_payments"].isna() | (working["total_sheduled_payments"] < 0),
        "total_sheduled_payments cannot be negative",
    )

    _append_reason(
        reasons,
        working["num_missed_payments"] > working["total_sheduled_payments"],
        "num_missed_payments cannot exceed total_sheduled_payments",
    )
    _append_reason(
        reasons,
        working["num_30_plus_late"] > working["total_sheduled_payments"],
        "num_30_plus_late cannot exceed total_sheduled_payments",
    )

    _append_reason(reasons, _is_blank(working["city"]), "city is required")
    _append_reason(reasons, _is_blank(working["state"]), "state is required")

    _append_reason(
        reasons,
        _is_blank(working["gender"]) | ~working["gender"].isin(VALID_GENDERS),
        "gender must match allowed values",
    )
    _append_reason(
        reasons,
        _is_blank(working["marital_status"])
        | ~working["marital_status"].isin(VALID_MARITAL_STATUSES),
        "marital_status must match allowed values",
    )
    _append_reason(
        reasons,
        _is_blank(working["employment_status"])
        | ~working["employment_status"].isin(VALID_EMPLOYMENT_STATUSES),
        "employment_status must match allowed values",
    )
    _append_reason(
        reasons,
        _is_blank(working["loan_type"]) | ~working["loan_type"].isin(VALID_LOAN_TYPES),
        "loan_type must match allowed values",
    )

    if require_target:
        _append_reason(
            reasons,
            working["default_flag"].isna() | ~working["default_flag"].isin([0, 1]),
            "default_flag must be 0 or 1",
        )

    rejected_mask = reasons.ne("")
    valid_df = working.loc[~rejected_mask].copy()
    rejected_df = working.loc[rejected_mask].copy()
    rejected_df["rejection_reasons"] = reasons.loc[rejected_mask]

    summary = {
        "total_rows": int(len(working)),
        "valid_rows": int(len(valid_df)),
        "rejected_rows": int(len(rejected_df)),
        "rejection_breakdown": _reason_breakdown(rejected_df),
    }
    return valid_df, rejected_df, summary


def validate_training_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _validate_required_columns(df, TRAINING_REQUIRED_COLUMNS)
    return _validate_rows(df, require_target=True)


def validate_scoring_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _validate_required_columns(df, SCORING_REQUIRED_COLUMNS)
    return _validate_rows(df, require_target=False)
