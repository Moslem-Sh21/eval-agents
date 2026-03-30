"""Environment variable configuration for the Fraud Analytics implementation.

Follows the same pydantic-settings pattern used by report_generation and
aml_investigation in this repo. All settings have sensible defaults so the
module can be imported without a .env file during tests.

Key difference from other modules: the dataset is pre-loaded inside an E2B
sandbox template (`DEFAULT_CODE_INTERPRETER_TEMPLATE`). No Kaggle credentials
are required — `cli.py create-db` pulls the CSVs from the sandbox filesystem.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _QuerySettings(BaseSettings):
    """Database query sub-settings."""

    model_config = SettingsConfigDict(env_prefix="FRAUD_DB__QUERY__", extra="ignore")

    mode: str = Field(default="ro", description="Query mode. 'ro' = read-only (enforced by SQL tool).")


class FraudDbSettings(BaseSettings):
    """SQLite database connection settings for the fraud transactions DB."""

    model_config = SettingsConfigDict(env_prefix="FRAUD_DB__", extra="ignore")

    driver: str = Field(default="sqlite", description="Database driver.")
    database: str = Field(
        default="implementations/fraud_analytics/data/fraud_transactions.db",
        description="Path to the SQLite database file, relative to repo root.",
    )
    query: _QuerySettings = Field(default_factory=_QuerySettings)


class FraudAnalyticsSettings(BaseSettings):
    """Top-level settings for the Fraud Analytics agent."""

    model_config = SettingsConfigDict(
        env_prefix="FRAUD_",
        extra="ignore",
        populate_by_name=True,  # lets e2b_template_id resolve from DEFAULT_CODE_INTERPRETER_TEMPLATE
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: Path = Field(
        default=Path("implementations/fraud_analytics/data"),
        description="Directory containing CSVs (pulled from E2B), DB, and case files.",
    )
    cases_path: Path = Field(
        default=Path("implementations/fraud_analytics/data/fraud_cases.jsonl"),
        description="Path to the generated JSONL case file.",
    )
    cases_with_output_path: Path = Field(
        default=Path("implementations/fraud_analytics/data/fraud_cases_with_output.jsonl"),
        description="Path written by the batch runner after agent runs.",
    )

    # Local copies of CSVs (downloaded from E2B template by cli.py create-db)
    transactions_csv: Path = Field(
        default=Path("implementations/fraud_analytics/data/transactions_data.csv"),
    )
    cards_csv: Path = Field(
        default=Path("implementations/fraud_analytics/data/cards_data.csv"),
    )
    users_csv: Path = Field(
        default=Path("implementations/fraud_analytics/data/users_data.csv"),
    )
    mcc_codes_json: Path = Field(
        default=Path("implementations/fraud_analytics/data/mcc_codes.json"),
    )

    # ------------------------------------------------------------------
    # E2B — dataset is pre-loaded in the template at /data/
    # ------------------------------------------------------------------
    e2b_api_key: str = Field(
        default="",
        description=(
            "E2B API key. Can also be set as the standard E2B_API_KEY env var — "
            "the E2B SDK picks that up automatically."
        ),
    )
    e2b_template_id: str = Field(
        default="q1sg157kmhnqbfjth0ue",
        description=(
            "E2B sandbox template ID. The template was built with the full Kaggle "
            "transactions-fraud-datasets pre-loaded at /data/. "
            "Set DEFAULT_CODE_INTERPRETER_TEMPLATE in .env to override."
        ),
        # pydantic-settings resolves env var DEFAULT_CODE_INTERPRETER_TEMPLATE
        # because we set populate_by_name=True and provide this alias.
        alias="default_code_interpreter_template",
    )

    # ------------------------------------------------------------------
    # Agent / model config — reuse global Gemini keys from .env
    # ------------------------------------------------------------------
    agent_model: str = Field(
        default="gemini-2.5-pro",
        description="Gemini model for the fraud agent worker.",
    )
    evaluator_model: str = Field(
        default="gemini-2.5-pro",
        description="Gemini model for the LLM-as-judge evaluator.",
    )

    # ------------------------------------------------------------------
    # Batch / eval knobs
    # ------------------------------------------------------------------
    agent_timeout: int = Field(default=300, description="Per-case agent timeout in seconds.")
    max_concurrent_cases: int = Field(default=5, description="Max concurrent cases in batch runner.")
    num_cases: int = Field(default=50, description="Number of cases to generate via create-cases.")
    fraud_ratio: float = Field(
        default=0.4,
        description="Fraction of generated cases that should seed a fraudulent transaction.",
    )

    db: FraudDbSettings = Field(default_factory=FraudDbSettings)


# Module-level singletons — import these everywhere.
settings = FraudAnalyticsSettings()
db_settings = FraudDbSettings()
