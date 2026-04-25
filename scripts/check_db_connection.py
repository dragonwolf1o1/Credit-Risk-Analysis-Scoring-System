"""Check the MySQL connection and key object counts."""

import pandas as pd

from credit_risk.config import settings
from credit_risk.db import get_engine


def main() -> int:
    engine = get_engine()
    with engine.connect():
        print("Connection to MySQL is successful.")

    counts = pd.read_sql(
        f"""
        SELECT
            (SELECT COUNT(*) FROM customers) AS customers_count,
            (SELECT COUNT(*) FROM loan) AS loan_count,
            (SELECT COUNT(*) FROM repayments) AS repayment_count,
            (SELECT COUNT(*) FROM {settings.training_view}) AS model_rows
        """,
        engine,
    )
    print(counts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
