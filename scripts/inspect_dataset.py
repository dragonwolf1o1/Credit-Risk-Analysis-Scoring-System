"""Preview the training dataset with validation summaries."""

import pandas as pd

from credit_risk.config import settings
from credit_risk.db import get_engine
from credit_risk.features import apply_feature_engineering
from credit_risk.validation import validate_training_dataframe


def main() -> int:
    query = f"SELECT * FROM {settings.training_view}"
    raw_df = pd.read_sql(query, get_engine())

    if raw_df.empty:
        print("No rows found in the training view.")
        return 0

    engineered_df = apply_feature_engineering(raw_df)
    valid_df, rejected_df, summary = validate_training_dataframe(engineered_df)

    print("Dataset summary:", summary)
    print("\nValidated sample:")
    print(valid_df.head())

    if not rejected_df.empty:
        print("\nRejected sample:")
        print(rejected_df[["loan_id", "customer_id", "rejection_reasons"]].head())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
