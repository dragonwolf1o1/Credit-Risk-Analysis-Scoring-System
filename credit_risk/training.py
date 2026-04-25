"""Training pipeline with validation, threshold tuning, and reports."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import settings
from .constants import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from .db import get_engine
from .features import apply_feature_engineering
from .thresholding import (
    calibrate_risk_band_thresholds,
    find_best_decision_threshold,
    summarize_risk_bands,
)
from .validation import validate_training_dataframe


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocess), ("model", model)])


def _load_training_dataset() -> pd.DataFrame:
    engine = get_engine()
    query = f"SELECT * FROM {settings.training_view}"
    return pd.read_sql(query, engine)


def _build_validation_strategy(y: pd.Series):
    class_counts = y.value_counts()
    if y.nunique() < 2:
        raise ValueError("Training dataset must contain both default and non-default loans.")

    min_class_count = int(class_counts.min())
    if len(y) < 6 or min_class_count < 2:
        return None

    n_splits = min(5, min_class_count)
    if n_splits < 2:
        return None

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def _evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "classification_report": _json_safe(report),
    }


def train_and_save_model() -> dict:
    settings.ensure_directories()

    raw_df = _load_training_dataset()
    if raw_df.empty:
        raise ValueError(
            f"No data found in {settings.training_view}. Add loan and repayment records first."
        )

    engineered_df = apply_feature_engineering(raw_df)
    valid_df, rejected_df, validation_summary = validate_training_dataframe(engineered_df)

    if rejected_df.empty:
        if settings.training_rejections_path.exists():
            settings.training_rejections_path.unlink()
    else:
        rejected_df.to_csv(settings.training_rejections_path, index=False)

    if valid_df.empty:
        raise ValueError("All training rows were rejected by validation rules.")

    X = valid_df[MODEL_FEATURES]
    y = valid_df["default_flag"].astype(int)

    pipeline = build_pipeline()
    validation_strategy = _build_validation_strategy(y)

    if validation_strategy is None:
        pipeline.fit(X, y)
        probabilities = pipeline.predict_proba(X)[:, 1]
        threshold_stats = find_best_decision_threshold(y, probabilities)
        risk_thresholds = calibrate_risk_band_thresholds(y, probabilities)
        evaluation_mode = "training_fallback"
    else:
        probabilities = cross_val_predict(
            build_pipeline(),
            X,
            y,
            cv=validation_strategy,
            method="predict_proba",
        )[:, 1]
        threshold_stats = find_best_decision_threshold(y, probabilities)
        risk_thresholds = calibrate_risk_band_thresholds(y, probabilities)
        pipeline.fit(X, y)
        evaluation_mode = "stratified_cross_validation"

    metrics = _evaluate_probabilities(y, probabilities, threshold_stats["threshold"])
    risk_band_summary = summarize_risk_bands(y, probabilities, risk_thresholds)

    bundle = {
        "pipeline": pipeline,
        "model_version": settings.model_version,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "decision_threshold": threshold_stats["threshold"],
        "decision_threshold_stats": threshold_stats,
        "risk_band_thresholds": risk_thresholds.as_dict(),
        "validation_mode": evaluation_mode,
        "metrics": metrics,
        "dataset_summary": validation_summary,
        "risk_band_summary": risk_band_summary,
    }

    joblib.dump(bundle, settings.model_path)

    report = {
        "model_artifact": str(settings.model_path),
        "model_version": settings.model_version,
        "trained_at": bundle["trained_at"],
        "validation_mode": evaluation_mode,
        "dataset_summary": validation_summary,
        "decision_threshold": bundle["decision_threshold"],
        "decision_threshold_stats": threshold_stats,
        "risk_band_thresholds": bundle["risk_band_thresholds"],
        "risk_band_summary": risk_band_summary,
        "metrics": metrics,
        "feature_columns": MODEL_FEATURES,
    }

    with settings.training_report_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2)

    return report


def print_training_summary(report: dict) -> None:
    dataset_summary = report["dataset_summary"]
    metrics = report["metrics"]

    print("Database data loaded and validated.")
    print(
        "Rows summary:",
        f"total={dataset_summary['total_rows']},",
        f"valid={dataset_summary['valid_rows']},",
        f"rejected={dataset_summary['rejected_rows']}",
    )
    print("Validation mode:", report["validation_mode"])
    print("Decision threshold:", report["decision_threshold"])
    print("Risk thresholds:", report["risk_band_thresholds"])
    print(
        "Metrics:",
        f"roc_auc={metrics['roc_auc']},",
        f"balanced_accuracy={metrics['balanced_accuracy']},",
        f"f1={metrics['f1']}",
    )
    print("Model saved to:", report["model_artifact"])
    print("Training report saved to:", settings.training_report_path)
    if dataset_summary["rejected_rows"]:
        print("Rejected rows saved to:", settings.training_rejections_path)
