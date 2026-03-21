-- Fraud Analytics SQLite schema
-- Matches the columns in computingvictor/transactions-fraud-datasets on Kaggle.
-- Tables: transactions, cards, users, mcc_codes
-- The is_fraud column is the ground-truth label (0 = legitimate, 1 = fraud).
-- All tables are populated by `cli.py create-db` from the downloaded CSVs.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- users
-- One row per customer. Sourced from users_data.csv.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id                  TEXT PRIMARY KEY,
    current_age         INTEGER,
    retirement_age      INTEGER,
    birth_year          INTEGER,
    birth_month         INTEGER,
    gender              TEXT,
    address             TEXT,
    latitude            REAL,
    longitude           REAL,
    per_capita_income   TEXT,   -- stored as string, e.g. "$45,000"
    yearly_income       TEXT,
    total_debt          TEXT,
    credit_score        INTEGER,
    num_credit_cards    INTEGER
);

-- ---------------------------------------------------------------------------
-- cards
-- One row per payment card. Sourced from cards_data.csv.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cards (
    id                  TEXT PRIMARY KEY,
    client_id           TEXT REFERENCES users(id),
    card_brand          TEXT,   -- Visa, Mastercard, Amex, Discover
    card_type           TEXT,   -- Credit, Debit, Debit (Prepaid)
    card_number         TEXT,
    expires             TEXT,   -- MM/YYYY
    cvv                 TEXT,
    has_chip            TEXT,   -- YES / NO
    cards_issued        INTEGER,
    credit_limit        TEXT,
    acct_open_date      TEXT,   -- MM/YYYY
    year_pin_last_changed INTEGER,
    card_on_dark_web    TEXT    -- YES / NO
);

-- ---------------------------------------------------------------------------
-- mcc_codes
-- Merchant Category Code lookup. Sourced from mcc_codes.json.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mcc_codes (
    mcc         INTEGER PRIMARY KEY,
    description TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- transactions
-- Core table. One row per transaction. Sourced from transactions_data.csv.
-- is_fraud is the ground-truth label used by check_accuracy().
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
    id                  TEXT PRIMARY KEY,
    date                TEXT NOT NULL,          -- ISO datetime string
    client_id           TEXT REFERENCES users(id),
    card_id             TEXT REFERENCES cards(id),
    amount              REAL NOT NULL,          -- transaction amount in USD
    use_chip            TEXT,                   -- "Chip Transaction", "Swipe Transaction", "Online Transaction"
    merchant_id         TEXT,
    merchant_city       TEXT,
    merchant_state      TEXT,
    zip                 TEXT,
    mcc                 INTEGER REFERENCES mcc_codes(mcc),
    errors              TEXT,                   -- NULL or pipe-separated error codes
    is_fraud            INTEGER NOT NULL DEFAULT 0  -- 0 = legitimate, 1 = fraudulent
);

-- ---------------------------------------------------------------------------
-- Indexes to speed up common agent queries
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_transactions_client    ON transactions(client_id);
CREATE INDEX IF NOT EXISTS idx_transactions_card      ON transactions(card_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date      ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud  ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant  ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_cards_client           ON cards(client_id);
