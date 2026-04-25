"""Database helpers."""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from .config import settings


def get_engine() -> Engine:
    return create_engine(settings.connection_url, pool_pre_ping=True)


def ensure_rejection_table(engine: Engine) -> None:
    statement = f"""
    CREATE TABLE IF NOT EXISTS {settings.rejection_table} (
        rejection_id INT AUTO_INCREMENT PRIMARY KEY,
        loan_id INT NOT NULL,
        customer_id INT NULL,
        rejection_stage VARCHAR(50) NOT NULL,
        rejection_reasons TEXT NOT NULL,
        source_script VARCHAR(100) NOT NULL,
        snapshot_payload LONGTEXT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_model_score_rejections_loan_id (loan_id)
    );
    """
    with engine.begin() as connection:
        connection.execute(text(statement))


def upsert_rejections(engine: Engine, rejected_df, stage: str, source_script: str) -> int:
    if rejected_df.empty:
        return 0

    ensure_rejection_table(engine)

    def optional_int(value):
        return None if pd.isna(value) else int(value)

    records = []
    for row in rejected_df.to_dict(orient="records"):
        records.append(
            {
                "loan_id": int(row.get("loan_id")),
                "customer_id": optional_int(row.get("customer_id")),
                "rejection_stage": stage,
                "rejection_reasons": row.get("rejection_reasons", "Unknown validation failure"),
                "source_script": source_script,
                "snapshot_payload": json.dumps(row, default=str),
            }
        )

    statement = text(
        f"""
        INSERT INTO {settings.rejection_table} (
            loan_id,
            customer_id,
            rejection_stage,
            rejection_reasons,
            source_script,
            snapshot_payload
        )
        VALUES (
            :loan_id,
            :customer_id,
            :rejection_stage,
            :rejection_reasons,
            :source_script,
            :snapshot_payload
        )
        ON DUPLICATE KEY UPDATE
            customer_id = VALUES(customer_id),
            rejection_stage = VALUES(rejection_stage),
            rejection_reasons = VALUES(rejection_reasons),
            source_script = VALUES(source_script),
            snapshot_payload = VALUES(snapshot_payload),
            updated_at = CURRENT_TIMESTAMP
        """
    )

    with engine.begin() as connection:
        connection.execute(statement, records)

    return len(records)


def clear_rejections(engine: Engine, loan_ids: Iterable[int]) -> int:
    loan_ids = [int(loan_id) for loan_id in loan_ids]
    if not loan_ids:
        return 0

    statement = text(
        f"DELETE FROM {settings.rejection_table} WHERE loan_id IN :loan_ids"
    ).bindparams(bindparam("loan_ids", expanding=True))

    with engine.begin() as connection:
        result = connection.execute(statement, {"loan_ids": loan_ids})

    return int(result.rowcount or 0)
