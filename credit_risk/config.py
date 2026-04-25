"""Project configuration and paths."""

from dataclasses import dataclass
from pathlib import Path
import os
from urllib.parse import quote_plus

from .constants import (
    DEFAULT_MODEL_VERSION,
    REJECTIONS_TABLE,
    SCORES_TABLE,
    TRAINING_DATASET_VIEW,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    db_user: str = os.getenv("CRS_DB_USER", "root")
    db_password_raw: str = os.getenv("CRS_DB_PASSWORD", "M4a1..,.,.,@")
    db_host: str = os.getenv("CRS_DB_HOST", "localhost")
    db_name: str = os.getenv("CRS_DB_NAME", "credit_risk_system")
    training_view: str = os.getenv("CRS_TRAINING_VIEW", TRAINING_DATASET_VIEW)
    scores_table: str = os.getenv("CRS_SCORES_TABLE", SCORES_TABLE)
    rejection_table: str = os.getenv("CRS_REJECTION_TABLE", REJECTIONS_TABLE)
    model_version: str = os.getenv("CRS_MODEL_VERSION", DEFAULT_MODEL_VERSION)
    poll_seconds: int = int(os.getenv("CRS_POLL_SECONDS", "10"))

    project_root: Path = PROJECT_ROOT
    model_path: Path = PROJECT_ROOT / "artifacts" / "models" / "credit_risk_model_bundle.joblib"
    training_report_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "training_report.json"
    training_rejections_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "training_rejections.csv"
    scoring_rejections_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "latest_scoring_rejections.csv"
    export_csv_path: Path = PROJECT_ROOT / "data" / "exports" / "excel_credit_risk.csv"
    export_xlsx_path: Path = PROJECT_ROOT / "data" / "exports" / "excel_credit_risk.xlsx"

    @property
    def connection_url(self) -> str:
        password = quote_plus(self.db_password_raw)
        return f"mysql+pymysql://{self.db_user}:{password}@{self.db_host}/{self.db_name}"

    def ensure_directories(self) -> None:
        for path in (
            self.model_path.parent,
            self.training_report_path.parent,
            self.export_csv_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
