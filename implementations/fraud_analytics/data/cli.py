"""CLI for setting up the Fraud Analytics data pipeline.

Commands
--------
create-db       Download the Kaggle dataset and build the SQLite database.
create-cases    Generate investigation case files (JSONL) from the DB.

Usage
-----
    uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-db
    uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-cases

Prerequisites
-------------
Set FRAUD_KAGGLE_USERNAME and FRAUD_KAGGLE_KEY in your .env, OR place a
kaggle.json at ~/.kaggle/kaggle.json before running create-db.
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
import pandas as pd

# Allow running as `python -m implementations.fraud_analytics.data.cli`
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from implementations.fraud_analytics.env_vars import settings
from implementations.fraud_analytics.models import CaseRecord, FraudPattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCHEMA_PATH = Path(__file__).parent / "schema.ddl"


def _infer_fraud_pattern(row: pd.Series) -> FraudPattern:
    """Heuristically assign a fraud pattern label from transaction features.

    This is used as soft ground-truth for pattern_type evaluation. It is NOT
    exposed to the agent — the agent must infer the pattern itself.
    """
    if not row.get("is_fraud", 0):
        return FraudPattern.NONE

    use_chip = str(row.get("use_chip", "")).lower()
    errors = str(row.get("errors", "") or "")
    amount = float(row.get("amount", 0) or 0)

    if "online" in use_chip:
        return FraudPattern.CARD_NOT_PRESENT
    if "insufficient" in errors.lower() or "bad pin" in errors.lower():
        return FraudPattern.ACCOUNT_TAKEOVER
    if amount > 500:
        return FraudPattern.UNUSUAL_VELOCITY
    if row.get("merchant_state") and row.get("merchant_state") != row.get("user_state", ""):
        return FraudPattern.GEO_ANOMALY
    return FraudPattern.UNKNOWN


def _get_db_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Fraud Analytics data pipeline commands."""


# ---------------------------------------------------------------------------
# create-db
# ---------------------------------------------------------------------------

@cli.command("create-db")
@click.option(
    "--db-path",
    default=str(settings.db.database),
    show_default=True,
    help="Output path for the SQLite database.",
)
@click.option(
    "--sample-size",
    default=100_000,
    show_default=True,
    help="Max rows to load from transactions_data.csv (use 0 for all rows).",
)
@click.option(
    "--skip-download",
    is_flag=True,
    default=False,
    help="Skip E2B pull if CSVs are already present locally.",
)
def create_db(db_path: str, sample_size: int, skip_download: bool) -> None:
    """Pull dataset from the pre-built E2B template and build the SQLite database.

    The E2B template (q1sg157kmhnqbfjth0ue) has the full Kaggle
    transactions-fraud-datasets pre-loaded at /data/. No Kaggle credentials
    are needed — only E2B_API_KEY is required.
    """
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Pull CSVs from E2B template sandbox ---------------------------
    if not skip_download:
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            click.echo(
                "❌ e2b_code_interpreter not installed. Run: uv add e2b-code-interpreter",
                err=True,
            )
            sys.exit(1)

        template_id = settings.e2b_template_id
        api_key = settings.e2b_api_key or None  # falls back to E2B_API_KEY env var

        click.echo(f"☁  Pulling dataset from E2B template '{template_id}'...")
        click.echo("   (Dataset is pre-loaded at /data/ in the sandbox — no Kaggle auth needed)")

        # Files to pull from the sandbox
        sandbox_files = {
            "transactions_data.csv": settings.transactions_csv,
            "cards_data.csv": settings.cards_csv,
            "users_data.csv": settings.users_csv,
            "mcc_codes.json": settings.mcc_codes_json,
        }

    sandbox = None
        try:
            sandbox = Sandbox.create(template=template_id, api_key=api_key)
            for sandbox_name, local_path in sandbox_files.items():
                sandbox_path = f"/data/{sandbox_name}"
                click.echo(f"  ⬇  Downloading {sandbox_path} → {local_path} ...")
                try:
                    content = sandbox.files.read(sandbox_path, format="bytes")
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(content)
                    size_mb = len(content) / 1_048_576
                    click.echo(f"     ✅ {size_mb:.1f} MB")
                except Exception as e:
                    if sandbox_name == "mcc_codes.json":
                        click.echo(f"     ⚠  {sandbox_name} not found in sandbox (optional): {e}")
                    else:
                        click.echo(f"     ❌ Failed to download {sandbox_name}: {e}", err=True)
                        sys.exit(1)
        except Exception as e:
            click.echo(f"❌ E2B sandbox error: {e}", err=True)
            click.echo(
                "Tip: ensure E2B_API_KEY is set in .env and the template ID is correct.\n"
                f"     Template: {template_id}",
                err=True,
            )
            sys.exit(1)
        finally:
            if sandbox is not None:
                sandbox.kill()
        click.echo("✅ All files pulled from E2B sandbox.")
    else:
        click.echo("⏭  Skipping E2B download (--skip-download set).")

    # --- 2. Load CSVs --------------------------------------------------------
    click.echo("📂 Loading CSV files...")

    transactions_csv = data_dir / "transactions_data.csv"
    cards_csv = data_dir / "cards_data.csv"
    users_csv = data_dir / "users_data.csv"
    mcc_json = data_dir / "mcc_codes.json"

    for path in [transactions_csv, cards_csv, users_csv]:
        if not path.exists():
            click.echo(f"❌ Expected file not found: {path}", err=True)
            sys.exit(1)

    # Load transactions (optionally sampled)
    click.echo(f"  Loading transactions (sample_size={sample_size or 'all'})...")
    tx_df = pd.read_csv(transactions_csv, low_memory=False)
    if sample_size and len(tx_df) > sample_size:
        # Stratified sample: preserve fraud ratio
        fraud_df = tx_df[tx_df["is_fraud"] == 1]
        legit_df = tx_df[tx_df["is_fraud"] == 0]
        fraud_n = min(len(fraud_df), max(1, int(sample_size * (len(fraud_df) / len(tx_df)))))
        legit_n = sample_size - fraud_n
        tx_df = pd.concat([
            fraud_df.sample(n=fraud_n, random_state=42),
            legit_df.sample(n=legit_n, random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        click.echo(f"  Sampled {len(tx_df):,} rows ({fraud_n:,} fraud, {legit_n:,} legit).")

    cards_df = pd.read_csv(cards_csv, low_memory=False)
    users_df = pd.read_csv(users_csv, low_memory=False)

    # --- 3. Normalise column names (snake_case, strip $, spaces) -------------
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
        return df

    tx_df = _norm(tx_df)
    cards_df = _norm(cards_df)
    users_df = _norm(users_df)

    # Ensure is_fraud is int 0/1 (some versions use 'Yes'/'No')
    if tx_df["is_fraud"].dtype == object:
        tx_df["is_fraud"] = tx_df["is_fraud"].str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
        ).fillna(0).astype(int)
    else:
        tx_df["is_fraud"] = tx_df["is_fraud"].fillna(0).astype(int)

    # Rename id columns to canonical names
    if "unnamed_0" in tx_df.columns:
        tx_df = tx_df.drop(columns=["unnamed_0"])

    # --- 4. Create / populate DB ---------------------------------------------
    click.echo(f"🗄  Creating database at {db}...")
    if db.exists():
        db.unlink()

    conn = sqlite3.connect(db)
    try:
        # Apply schema
        schema_sql = SCHEMA_PATH.read_text()
        conn.executescript(schema_sql)

        # Load mcc_codes
        if mcc_json.exists():
            click.echo("  Loading MCC codes...")
            mcc_data = json.loads(mcc_json.read_text())
            if isinstance(mcc_data, dict):
                mcc_rows = [(int(k), v) for k, v in mcc_data.items() if k.isdigit()]
            else:
                mcc_rows = [(int(r["mcc"]), r.get("description", "")) for r in mcc_data]
            conn.executemany(
                "INSERT OR IGNORE INTO mcc_codes(mcc, description) VALUES (?, ?)",
                mcc_rows,
            )
        else:
            click.echo("  ⚠  mcc_codes.json not found — mcc_codes table will be empty.")

        # Load users
        click.echo(f"  Loading {len(users_df):,} users...")
        users_df.to_sql("users", conn, if_exists="replace", index=False)

        # Load cards
        click.echo(f"  Loading {len(cards_df):,} cards...")
        cards_df.to_sql("cards", conn, if_exists="replace", index=False)

        # Load transactions
        click.echo(f"  Loading {len(tx_df):,} transactions...")
        tx_df.to_sql("transactions", conn, if_exists="replace", index=False)

        conn.commit()
    finally:
        conn.close()

    click.echo(f"✅ Database created: {db}")
    click.echo(f"   Transactions: {len(tx_df):,} ({tx_df['is_fraud'].sum():,} fraudulent)")


# ---------------------------------------------------------------------------
# create-cases
# ---------------------------------------------------------------------------

@cli.command("create-cases")
@click.option(
    "--db-path",
    default=str(settings.db.database),
    show_default=True,
    help="Path to the SQLite database.",
)
@click.option(
    "--output-path",
    default=str(settings.cases_path),
    show_default=True,
    help="Output JSONL file for case records.",
)
@click.option(
    "--num-cases",
    default=settings.num_cases,
    show_default=True,
    help="Total number of cases to generate.",
)
@click.option(
    "--fraud-ratio",
    default=settings.fraud_ratio,
    show_default=True,
    help="Fraction of cases that seed a fraudulent transaction.",
)
@click.option(
    "--window-days",
    default=30,
    show_default=True,
    help="Number of days before seed transaction to include in the investigation window.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for reproducible case selection.",
)
def create_cases(
    db_path: str,
    output_path: str,
    num_cases: int,
    fraud_ratio: float,
    window_days: int,
    seed: int,
) -> None:
    """Generate investigation case files from the database."""
    db = Path(db_path)
    if not db.exists():
        click.echo(f"❌ Database not found: {db}. Run create-db first.", err=True)
        sys.exit(1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    conn = _get_db_connection(db)

    try:
        # Sample fraud transactions
        n_fraud = max(1, int(num_cases * fraud_ratio))
        n_legit = num_cases - n_fraud

        fraud_rows = conn.execute(
            f"SELECT id, date, client_id, card_id, amount, use_chip, merchant_state, "
            f"errors, mcc, is_fraud FROM transactions WHERE is_fraud = 1 "
            f"ORDER BY RANDOM() LIMIT {n_fraud}"
        ).fetchall()

        legit_rows = conn.execute(
            f"SELECT id, date, client_id, card_id, amount, use_chip, merchant_state, "
            f"errors, mcc, is_fraud FROM transactions WHERE is_fraud = 0 "
            f"ORDER BY RANDOM() LIMIT {n_legit}"
        ).fetchall()

        all_rows = list(fraud_rows) + list(legit_rows)
        random.shuffle(all_rows)

        if len(all_rows) == 0:
            click.echo("❌ No transactions found in DB. Run create-db first.", err=True)
            sys.exit(1)

        click.echo(f"📋 Generating {len(all_rows)} cases ({n_fraud} fraud, {n_legit} legit)...")

        trigger_labels_fraud = [
            "High-amount transaction alert",
            "Geo-velocity anomaly detected",
            "Multiple declined transactions",
            "First transaction in new merchant category",
            "Transaction outside normal hours",
            "Chip bypass attempt flagged",
            "Velocity rule triggered (3+ txn/hour)",
        ]
        trigger_labels_legit = [
            "Random sampling review",
            "Periodic compliance audit",
            "Customer dispute review",
            "Model calibration sample",
        ]

        cases: list[CaseRecord] = []
        for i, row in enumerate(all_rows):
            row_dict = dict(row)
            is_fraud = bool(row_dict.get("is_fraud", 0))

            # Compute investigation window
            try:
                seed_dt = datetime.fromisoformat(str(row_dict["date"]))
            except (ValueError, TypeError):
                seed_dt = datetime(2020, 1, 1)

            window_start = (seed_dt - timedelta(days=window_days)).isoformat()

            # Infer fraud pattern
            pattern = _infer_fraud_pattern(pd.Series(row_dict))

            trigger = (
                random.choice(trigger_labels_fraud) if is_fraud
                else random.choice(trigger_labels_legit)
            )

            case = CaseRecord(
                case_id=f"CASE_{i + 1:04d}",
                seed_transaction_id=str(row_dict["id"]),
                client_id=str(row_dict.get("client_id", "")),
                card_id=str(row_dict.get("card_id", "")),
                window_start=window_start,
                window_end=seed_dt.isoformat(),
                trigger_label=trigger,
                ground_truth_is_fraud=is_fraud,
                ground_truth_pattern=pattern,
            )
            cases.append(case)

    finally:
        conn.close()

    # Write JSONL
    with open(output, "w") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")

    fraud_count = sum(1 for c in cases if c.ground_truth_is_fraud)
    click.echo(f"✅ Wrote {len(cases)} cases to {output}")
    click.echo(f"   Fraud: {fraud_count} | Legit: {len(cases) - fraud_count}")


if __name__ == "__main__":
    cli()
