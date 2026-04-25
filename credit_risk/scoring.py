"""Scoring workflow with validation and rejection logging."""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from .config import settings
from .constants import DEFAULT_RISK_BAND_THRESHOLDS, MODEL_FEATURES
from .db import clear_rejections, ensure_rejection_table, get_engine, upsert_rejections
from .features import apply_feature_engineering
from .thresholding import get_risk_band
from .validation import validate_scoring_dataframe


def load_model_bundle() -> dict:
    bundle = joblib.load(settings.model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            f"Model artifact at {settings.model_path} is not a valid model bundle."
        )
    return bundle


def fetch_pending_loans(engine) -> pd.DataFrame:
    query = f"""
    SELECT v.*
    FROM {settings.training_view} v
    LEFT JOIN {settings.scores_table} s
        ON v.loan_id = s.loan_id
    WHERE s.loan_id IS NULL;
    """
    return pd.read_sql(query, engine)


def _save_latest_rejections(rejected_df: pd.DataFrame) -> None:
    if rejected_df.empty:
        if settings.scoring_rejections_path.exists():
            settings.scoring_rejections_path.unlink()
        return
    rejected_df.to_csv(settings.scoring_rejections_path, index=False)


def score_dataframe(df: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    engineered_df = apply_feature_engineering(df)
    valid_df, rejected_df, validation_summary = validate_scoring_dataframe(engineered_df)

    if valid_df.empty:
        return pd.DataFrame(), rejected_df, validation_summary

    pipeline = bundle["pipeline"]
    risk_thresholds = bundle.get("risk_band_thresholds", DEFAULT_RISK_BAND_THRESHOLDS)
    model_version = bundle.get("model_version", settings.model_version)

    probabilities = pipeline.predict_proba(valid_df[MODEL_FEATURES])[:, 1]
    scored_df = valid_df[["loan_id", "customer_id"]].copy()
    scored_df["prob_default"] = [round(float(probability), 4) for probability in probabilities]
    scored_df["risk_band"] = [
        get_risk_band(float(probability), risk_thresholds) for probability in probabilities
    ]
    scored_df["model_version"] = model_version

    return scored_df, rejected_df, validation_summary


def score_pending_loans(source_script: str = "score_loan_once") -> dict:
    settings.ensure_directories()

    engine = get_engine()
    ensure_rejection_table(engine)

    pending_df = fetch_pending_loans(engine)

    if pending_df.empty:
        return {
            "pending_rows": 0,
            "scored_rows": 0,
            "rejected_rows": 0,
            "message": "No new loans found to score.",
        }

    bundle = load_model_bundle()

    scored_df, rejected_df, validation_summary = score_dataframe(pending_df, bundle)

    if not rejected_df.empty:
        upsert_rejections(engine, rejected_df, "scoring_validation", source_script)
    _save_latest_rejections(rejected_df)

    if scored_df.empty:
        return {
            "pending_rows": int(len(pending_df)),
            "scored_rows": 0,
            "rejected_rows": int(len(rejected_df)),
            "message": "All pending loans failed validation.",
            "validation_summary": validation_summary,
        }

    scored_df.to_sql(settings.scores_table, engine, if_exists="append", index=False)
    cleared_rejections = clear_rejections(engine, scored_df["loan_id"].tolist())

    return {
        "pending_rows": int(len(pending_df)),
        "scored_rows": int(len(scored_df)),
        "rejected_rows": int(len(rejected_df)),
        "cleared_rejections": int(cleared_rejections),
        "message": "Pending loans scored successfully.",
        "validation_summary": validation_summary,
        "scores_preview": scored_df.to_dict(orient="records"),
    }


def print_scoring_summary(result: dict) -> None:
    print(result["message"])
    print(
        "Rows summary:",
        f"pending={result['pending_rows']},",
        f"scored={result['scored_rows']},",
        f"rejected={result['rejected_rows']}",
    )
    if result.get("cleared_rejections"):
        print("Cleared rejection records:", result["cleared_rejections"])
    if result.get("validation_summary"):
        print("Validation summary:", json.dumps(result["validation_summary"], indent=2))
    if result.get("scores_preview"):
        print("Preview of scored rows:")
        print(pd.DataFrame(result["scores_preview"]).head())


def run_daemon() -> None:
    print("Starting auto-scoring daemon.")
    print("Press CTRL + C to stop.\n")

    while True:
        try:
            result = score_pending_loans(source_script="score_loan_daemon")
            print_scoring_summary(result)
            print(f"Waiting {settings.poll_seconds} seconds...\n")
            time.sleep(settings.poll_seconds)
        except KeyboardInterrupt:
            print("\nAuto-scoring stopped manually.")
            break
        except Exception as error:
            print(f"Error: {error}")
            print("Retrying in 5 seconds...\n")
            time.sleep(5)
