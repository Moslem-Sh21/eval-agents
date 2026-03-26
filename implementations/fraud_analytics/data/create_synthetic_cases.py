"""Synthetic Fraud Case Generator — fixed version.

Inserts synthetic transactions directly into fraud_transactions.db
and generates synthetic_cases.jsonl for evaluation.

Users/cards inserts are skipped — the agent only queries the
transactions table, so no matching user/card rows are needed.

Usage:
    uv run --env-file .env python -m implementations.fraud_analytics.data.create_synthetic_cases
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from implementations.fraud_analytics.models import CaseRecord, FraudPattern

DB_PATH     = Path("implementations/fraud_analytics/data/fraud_transactions.db")
OUTPUT_PATH = Path("implementations/fraud_analytics/data/synthetic_cases.jsonl")
WINDOW_DAYS = 7


def txn_id() -> str:
    return f"SYN_TXN_{uuid.uuid4().hex[:8].upper()}"

def client_id(n: int) -> str:
    return f"SYN_CLIENT_{n:04d}"

def card_id(n: int) -> str:
    return f"SYN_CARD_{n:04d}"

def merchant_id(name: str) -> str:
    return f"SYN_MERCH_{name.upper().replace(' ', '_')[:12]}"


def insert_transactions(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany("""
        INSERT OR IGNORE INTO transactions (
            transaction_id, client_id, card_id, amount, use_chip,
            merchant_id, merchant_city, merchant_state, zip, mcc,
            errors, is_fraud, date
        ) VALUES (
            :transaction_id, :client_id, :card_id, :amount, :use_chip,
            :merchant_id, :merchant_city, :merchant_state, :zip, :mcc,
            :errors, :is_fraud, :date
        )
    """, rows)
    conn.commit()


def normal_history(cid: int, anchor: datetime, n: int = 15) -> list[dict]:
    rows = []
    for i in range(n):
        dt = anchor - timedelta(days=i * 2, hours=i % 8)
        rows.append({
            "transaction_id": txn_id(),
            "client_id": client_id(cid),
            "card_id": card_id(cid),
            "amount": round(20 + (i % 5) * 15 + 0.99, 2),
            "use_chip": "Chip Transaction",
            "merchant_id": merchant_id(f"GROCERY_{i%3}"),
            "merchant_city": "New York", "merchant_state": "NY",
            "zip": "10001", "mcc": 5411,
            "errors": None, "is_fraud": 0,
            "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return rows


def scenario_card_not_present(conn, cid, anchor):
    history = normal_history(cid, anchor)
    seed_txn = txn_id()
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 847.99, "use_chip": "Online Transaction",
        "merchant_id": merchant_id("ELECTRONICS_ONLINE"),
        "merchant_city": "Online", "merchant_state": "CA", "zip": "90210", "mcc": 5732,
        "errors": None, "is_fraud": 1, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + [seed])
    return seed_txn, True, FraudPattern.CARD_NOT_PRESENT


def scenario_unusual_velocity(conn, cid, anchor):
    history = normal_history(cid, anchor)
    txns = []
    seed_txn = txn_id()
    for i in range(8):
        tid = seed_txn if i == 0 else txn_id()
        txns.append({
            "transaction_id": tid, "client_id": client_id(cid), "card_id": card_id(cid),
            "amount": round(19.99 + i * 2, 2), "use_chip": "Chip Transaction",
            "merchant_id": merchant_id(f"STORE_{i}"),
            "merchant_city": "New York", "merchant_state": "NY", "zip": "10001", "mcc": 5999,
            "errors": None, "is_fraud": 1,
            "date": (anchor + timedelta(minutes=i * 11)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    insert_transactions(conn, history + txns)
    return seed_txn, True, FraudPattern.UNUSUAL_VELOCITY


def scenario_account_takeover(conn, cid, anchor):
    history = normal_history(cid, anchor)
    seed_txn = txn_id()
    error_txns = []
    for i in range(3):
        error_txns.append({
            "transaction_id": txn_id(), "client_id": client_id(cid), "card_id": card_id(cid),
            "amount": 500.0, "use_chip": "Chip Transaction",
            "merchant_id": merchant_id("ATM_DOWNTOWN"),
            "merchant_city": "New York", "merchant_state": "NY", "zip": "10001", "mcc": 6011,
            "errors": "Bad PIN", "is_fraud": 1,
            "date": (anchor - timedelta(minutes=15 - i * 4)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 1200.0, "use_chip": "Chip Transaction",
        "merchant_id": merchant_id("ATM_DOWNTOWN"),
        "merchant_city": "New York", "merchant_state": "NY", "zip": "10001", "mcc": 6011,
        "errors": None, "is_fraud": 1, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + error_txns + [seed])
    return seed_txn, True, FraudPattern.ACCOUNT_TAKEOVER


def scenario_identity_theft(conn, cid, anchor):
    history = normal_history(cid, anchor)
    seed_txn = txn_id()
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 3200.00, "use_chip": "Chip Transaction",
        "merchant_id": merchant_id("FINE_JEWELLERY"),
        "merchant_city": "Beverly Hills", "merchant_state": "CA", "zip": "90210", "mcc": 5944,
        "errors": None, "is_fraud": 1, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + [seed])
    return seed_txn, True, FraudPattern.IDENTITY_THEFT


def scenario_smurfing(conn, cid, anchor):
    history = normal_history(cid, anchor)
    txns = []
    seed_txn = txn_id()
    amounts = [490.00, 493.50, 497.00, 494.75, 498.99, 491.25]
    for i, amt in enumerate(amounts):
        tid = seed_txn if i == 0 else txn_id()
        txns.append({
            "transaction_id": tid, "client_id": client_id(cid), "card_id": card_id(cid),
            "amount": amt, "use_chip": "Online Transaction",
            "merchant_id": merchant_id(f"WIRE_SERVICE_{i}"),
            "merchant_city": "Online", "merchant_state": "TX", "zip": "75001", "mcc": 6051,
            "errors": None, "is_fraud": 1,
            "date": (anchor + timedelta(hours=i * 3)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    insert_transactions(conn, history + txns)
    return seed_txn, True, FraudPattern.SMURFING


def scenario_merchant_fraud(conn, cid, anchor):
    history = normal_history(cid, anchor)
    seed_txn = txn_id()
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 1750.00, "use_chip": "Chip Transaction",
        "merchant_id": "UNKNOWN_MERCH_ZZ99999",
        "merchant_city": "Las Vegas", "merchant_state": "NV", "zip": "89101", "mcc": 5999,
        "errors": None, "is_fraud": 1, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + [seed])
    return seed_txn, True, FraudPattern.MERCHANT_FRAUD


def scenario_geo_anomaly(conn, cid, anchor):
    history = normal_history(cid, anchor)
    nyc_txn = {
        "transaction_id": txn_id(), "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 45.00, "use_chip": "Chip Transaction",
        "merchant_id": merchant_id("NYC_COFFEE"),
        "merchant_city": "New York", "merchant_state": "NY", "zip": "10001", "mcc": 5812,
        "errors": None, "is_fraud": 0,
        "date": (anchor - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    seed_txn = txn_id()
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 620.00, "use_chip": "Chip Transaction",
        "merchant_id": merchant_id("LONDON_HOTEL"),
        "merchant_city": "London", "merchant_state": "ENG", "zip": "EC1A", "mcc": 7011,
        "errors": None, "is_fraud": 1, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + [nyc_txn, seed])
    return seed_txn, True, FraudPattern.GEO_ANOMALY


def scenario_unknown_pattern(conn, cid, anchor):
    history = normal_history(cid, anchor)
    seed_txn = txn_id()
    declined = {
        "transaction_id": txn_id(), "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 312.00, "use_chip": "Online Transaction",
        "merchant_id": merchant_id("RETAIL_MID"),
        "merchant_city": "Miami", "merchant_state": "FL", "zip": "33101", "mcc": 5411,
        "errors": "Insufficient funds", "is_fraud": 1,
        "date": (anchor - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 312.00, "use_chip": "Online Transaction",
        "merchant_id": merchant_id("RETAIL_MID"),
        "merchant_city": "Miami", "merchant_state": "FL", "zip": "33101", "mcc": 5411,
        "errors": None, "is_fraud": 1, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + [declined, seed])
    return seed_txn, True, FraudPattern.UNKNOWN


def scenario_legit_high_amount(conn, cid, anchor):
    history = []
    for i in range(10):
        history.append({
            "transaction_id": txn_id(), "client_id": client_id(cid), "card_id": card_id(cid),
            "amount": round(400 + (i % 4) * 200 + 0.99, 2), "use_chip": "Chip Transaction",
            "merchant_id": merchant_id(f"DEPT_STORE_{i%2}"),
            "merchant_city": "New York", "merchant_state": "NY", "zip": "10001", "mcc": 5311,
            "errors": None, "is_fraud": 0,
            "date": (anchor - timedelta(days=i * 3)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    seed_txn = txn_id()
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 1100.00, "use_chip": "Chip Transaction",
        "merchant_id": merchant_id("TRAVEL_AGENCY"),
        "merchant_city": "New York", "merchant_state": "NY", "zip": "10001", "mcc": 4722,
        "errors": None, "is_fraud": 0, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + [seed])
    return seed_txn, False, FraudPattern.NONE


def scenario_legit_new_merchant(conn, cid, anchor):
    history = normal_history(cid, anchor)
    seed_txn = txn_id()
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 34.50, "use_chip": "Chip Transaction",
        "merchant_id": merchant_id("NEW_RESTAURANT_A"),
        "merchant_city": "New York", "merchant_state": "NY", "zip": "10001", "mcc": 5812,
        "errors": None, "is_fraud": 0, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + [seed])
    return seed_txn, False, FraudPattern.NONE


def scenario_legit_velocity(conn, cid, anchor):
    history = normal_history(cid, anchor)
    txns = []
    seed_txn = txn_id()
    for i in range(5):
        tid = seed_txn if i == 0 else txn_id()
        txns.append({
            "transaction_id": tid, "client_id": client_id(cid), "card_id": card_id(cid),
            "amount": round(12 + i * 8 + 0.49, 2), "use_chip": "Chip Transaction",
            "merchant_id": merchant_id("SUPERMARKET_A"),
            "merchant_city": "New York", "merchant_state": "NY", "zip": "10001", "mcc": 5411,
            "errors": None, "is_fraud": 0,
            "date": (anchor + timedelta(minutes=i * 22)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    insert_transactions(conn, history + txns)
    return seed_txn, False, FraudPattern.NONE


def scenario_legit_online(conn, cid, anchor):
    history = []
    for i in range(12):
        history.append({
            "transaction_id": txn_id(), "client_id": client_id(cid), "card_id": card_id(cid),
            "amount": round(25 + (i % 6) * 10 + 0.99, 2), "use_chip": "Online Transaction",
            "merchant_id": merchant_id(f"ONLINE_RETAIL_{i%4}"),
            "merchant_city": "Online", "merchant_state": "CA", "zip": "90210", "mcc": 5732,
            "errors": None, "is_fraud": 0,
            "date": (anchor - timedelta(days=i * 2)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    seed_txn = txn_id()
    seed = {
        "transaction_id": seed_txn, "client_id": client_id(cid), "card_id": card_id(cid),
        "amount": 89.99, "use_chip": "Online Transaction",
        "merchant_id": merchant_id("ONLINE_RETAIL_5"),
        "merchant_city": "Online", "merchant_state": "CA", "zip": "90210", "mcc": 5732,
        "errors": None, "is_fraud": 0, "date": anchor.strftime("%Y-%m-%d %H:%M:%S"),
    }
    insert_transactions(conn, history + [seed])
    return seed_txn, False, FraudPattern.NONE


SCENARIOS = [
    (scenario_card_not_present, 1,  "card_not_present — online, no chip, high amount"),
    (scenario_unusual_velocity, 2,  "unusual_velocity — 8 txns in 90 minutes"),
    (scenario_account_takeover, 3,  "account_takeover — bad PIN then large withdrawal"),
    (scenario_identity_theft,   4,  "identity_theft — never-seen MCC, high amount"),
    (scenario_smurfing,         5,  "smurfing — 6 txns just under $500"),
    (scenario_merchant_fraud,   6,  "merchant_fraud — unknown merchant ID"),
    (scenario_geo_anomaly,      7,  "geo_anomaly — NYC then London 2 hrs apart"),
    (scenario_unknown_pattern,  8,  "unknown — multiple weak signals"),
    (scenario_legit_high_amount,9,  "legit_high_amount — large but consistent with history"),
    (scenario_legit_new_merchant,10,"legit_new_merchant — new merchant, normal amount"),
    (scenario_legit_velocity,   11, "legit_velocity — 5 txns, same merchant, shopping trip"),
    (scenario_legit_online,     12, "legit_online — online but client shops online regularly"),
]


def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. Run create-db first.")
        return

    conn = sqlite3.connect(DB_PATH)
    anchor = datetime(2024, 3, 15, 14, 30, 0)
    cases: list[CaseRecord] = []

    print(f"\n🧪 Generating {len(SCENARIOS)} synthetic cases...\n")

    for i, (fn, cid, description) in enumerate(SCENARIOS, start=1):
        seed_txn, is_fraud, pattern = fn(conn, cid, anchor)
        case = CaseRecord(
            case_id=f"SYN_{i:04d}",
            seed_transaction_id=seed_txn,
            client_id=client_id(cid),
            card_id=card_id(cid),
            window_start=(anchor - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d"),
            window_end=(anchor + timedelta(days=1)).strftime("%Y-%m-%d"),
            trigger_label=description,
            ground_truth_is_fraud=is_fraud,
            ground_truth_pattern=pattern,
        )
        cases.append(case)
        label = "FRAUD" if is_fraud else "LEGIT"
        print(f"  SYN_{i:04d}: [{label}] {description}")

    conn.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")

    fraud_count = sum(1 for c in cases if c.ground_truth_is_fraud)
    legit_count = sum(1 for c in cases if not c.ground_truth_is_fraud)
    print(f"\n✅ Written {len(cases)} synthetic cases to {OUTPUT_PATH}")
    print(f"   Fraud: {fraud_count} | Legit: {legit_count}")
    print(f"\n📊 Run evaluation with:")
    print(f"   uv run --env-file .env python -m implementations.fraud_analytics.evaluate \\")
    print(f"     --dataset-path {OUTPUT_PATH} \\")
    print(f"     --dataset-name fraud-analytics-synthetic \\")
    print(f"     --max-concurrent-cases 3")


if __name__ == "__main__":
    main()
