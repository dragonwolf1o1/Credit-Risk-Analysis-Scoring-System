"""Export scored loans to CSV and Excel."""

import pandas as pd

from credit_risk.config import settings
from credit_risk.db import get_engine


def main() -> int:
    settings.ensure_directories()
    query = f"""
    SELECT
        v.loan_id,
        v.customer_id,
        v.full_name,
        v.gender,
        v.marital_status,
        v.employment_status,
        v.annual_income,
        v.city,
        v.state,
        v.loan_amount,
        v.intrest_rate,
        v.loan_terms_month,
        v.loan_type,
        v.colletral_type,
        s.prob_default,
        s.risk_band,
        s.prediction_date,
        s.model_version
    FROM {settings.training_view} v
    JOIN {settings.scores_table} s
        ON v.loan_id = s.loan_id;
    """

    df = pd.read_sql(query, get_engine())
    df.to_csv(settings.export_csv_path, index=False)

    try:
        df.to_excel(settings.export_xlsx_path, index=False)
        print("CSV and Excel exports created successfully.")
        print("CSV:", settings.export_csv_path)
        print("Excel:", settings.export_xlsx_path)
    except Exception as error:
        print("CSV export created successfully.")
        print("CSV:", settings.export_csv_path)
        print("Excel export skipped:", error)

    print("Rows exported:", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
