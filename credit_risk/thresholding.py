"""Probability threshold tuning and risk band calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from .constants import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RISK_BAND_THRESHOLDS,
    RISK_BANDS,
    TARGET_BAND_DEFAULT_RATES,
)


@dataclass(frozen=True)
class RiskBandThresholds:
    low_max: float = DEFAULT_RISK_BAND_THRESHOLDS["low_max"]
    medium_max: float = DEFAULT_RISK_BAND_THRESHOLDS["medium_max"]
    high_max: float = DEFAULT_RISK_BAND_THRESHOLDS["high_max"]

    def normalized(self) -> "RiskBandThresholds":
        values = [self.low_max, self.medium_max, self.high_max]
        bounded = [min(max(value, 0.01), 0.99) for value in values]
        for index in range(1, len(bounded)):
            if bounded[index] <= bounded[index - 1]:
                bounded[index] = min(bounded[index - 1] + 0.01, 0.99)
        return RiskBandThresholds(*bounded)

    def as_dict(self) -> dict[str, float]:
        return asdict(self.normalized())


def get_risk_band(probability: float, thresholds: dict[str, float] | RiskBandThresholds) -> str:
    if isinstance(thresholds, RiskBandThresholds):
        thresholds = thresholds.as_dict()

    if probability < thresholds["low_max"]:
        return "Low"
    if probability < thresholds["medium_max"]:
        return "Medium"
    if probability < thresholds["high_max"]:
        return "High"
    return "Very High"


def assign_risk_bands(probabilities, thresholds: dict[str, float] | RiskBandThresholds) -> pd.Series:
    return pd.Series([get_risk_band(float(probability), thresholds) for probability in probabilities])


def find_best_decision_threshold(y_true, probabilities) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    best = {
        "threshold": DEFAULT_DECISION_THRESHOLD,
        "balanced_accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(int)
        stats = {
            "threshold": round(float(threshold), 4),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(y_true, predictions)), 4
            ),
            "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
            "precision": round(
                float(precision_score(y_true, predictions, zero_division=0)), 4
            ),
            "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        }
        current_score = (
            stats["balanced_accuracy"],
            stats["f1"],
            stats["recall"],
            -abs(stats["threshold"] - DEFAULT_DECISION_THRESHOLD),
        )
        best_score = (
            best["balanced_accuracy"],
            best["f1"],
            best["recall"],
            -abs(best["threshold"] - DEFAULT_DECISION_THRESHOLD),
        )
        if current_score > best_score:
            best = stats

    return best


def _fallback_band_thresholds(probabilities) -> RiskBandThresholds:
    quantiles = np.quantile(np.asarray(probabilities, dtype=float), [0.25, 0.50, 0.75])
    return RiskBandThresholds(
        low_max=float(quantiles[0]),
        medium_max=float(quantiles[1]),
        high_max=float(quantiles[2]),
    ).normalized()


def calibrate_risk_band_thresholds(y_true, probabilities) -> RiskBandThresholds:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(probabilities) < 4 or len(np.unique(probabilities)) < 4:
        return _fallback_band_thresholds(probabilities)

    quantile_points = np.linspace(0.15, 0.85, 8)
    candidates = sorted(
        {
            round(float(value), 4)
            for value in np.quantile(probabilities, quantile_points)
            if 0.01 < float(value) < 0.99
        }
    )
    if len(candidates) < 3:
        return _fallback_band_thresholds(probabilities)

    best_thresholds = None
    best_score = None

    for low_max, medium_max, high_max in combinations(candidates, 3):
        thresholds = RiskBandThresholds(low_max, medium_max, high_max).normalized()
        bands = assign_risk_bands(probabilities, thresholds)

        counts = []
        default_rates = []
        valid = True
        for band in RISK_BANDS:
            mask = bands == band
            count = int(mask.sum())
            if count == 0:
                valid = False
                break
            counts.append(count)
            default_rates.append(float(y_true[mask.values].mean()))

        if not valid:
            continue
        if any(left > right for left, right in zip(default_rates, default_rates[1:])):
            continue

        penalty = 0.0
        for band, observed_rate, count in zip(RISK_BANDS, default_rates, counts):
            target_rate = TARGET_BAND_DEFAULT_RATES[band]
            penalty += (observed_rate - target_rate) ** 2
            penalty += 0.01 / count

        score = round(float(penalty), 8)
        if best_score is None or score < best_score:
            best_score = score
            best_thresholds = thresholds

    if best_thresholds is None:
        return _fallback_band_thresholds(probabilities)

    return best_thresholds


def summarize_risk_bands(y_true, probabilities, thresholds) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    bands = assign_risk_bands(probabilities, thresholds)

    summary = {}
    for band in RISK_BANDS:
        mask = bands == band
        count = int(mask.sum())
        default_rate = float(y_true[mask.values].mean()) if count else 0.0
        summary[band] = {
            "count": count,
            "default_rate": round(default_rate, 4),
        }
    return summary
