# Fraud Analytics Investigation Agent

A Google ADK agent that investigates financial transaction fraud cases by:
- Querying a SQLite database of real transactions via a **read-only SQL tool**
- Running deeper Python analysis in an **E2B sandbox**
- Checking ground-truth fraud labels via a **`check_accuracy` tool**
- Producing structured verdicts with fraud pattern classification

Built on the same patterns as `aml_investigation` and `report_generation` in this repo.

---

## Architecture

```
Data pipeline (data/cli.py)
  └─ Downloads Kaggle dataset → SQLite DB → fraud_cases.jsonl

Agent (agent.py)
  ├─ get_schema()        — inspect DB before querying
  ├─ execute_sql()       — read-only SQL (AST-level SELECT enforcement)
  ├─ check_accuracy()    — ground-truth label lookup by transaction ID
  └─ run_python()        — E2B secure Python sandbox

Demo UI (gradio_app.py)
  └─ NL query → agent verdict + accuracy score inline

Evaluation (evaluate.py)
  ├─ Item-level:  deterministic grader + LLM-as-judge (explanation quality)
  ├─ Trace-level: SQL safety · E2B exec success · check_accuracy called
  └─ Run-level:   precision / recall / F1 · confusion matrix → LangFuse
```

---

## Setup

### 1. Add environment variables

Append these to your `.env` file (copy from `.env.example` if needed):

```bash
# Fraud Analytics database
FRAUD_DB__DRIVER="sqlite"
FRAUD_DB__DATABASE="implementations/fraud_analytics/data/fraud_transactions.db"
FRAUD_DB__QUERY__MODE="ro"

# E2B — dataset is pre-loaded in the template, no Kaggle credentials needed
E2B_API_KEY="e2b_..."
DEFAULT_CODE_INTERPRETER_TEMPLATE="q1sg157kmhnqbfjth0ue"

# Optional model overrides (defaults are sensible)
# FRAUD_AGENT_MODEL="gemini-2.5-flash"
# FRAUD_EVALUATOR_MODEL="gemini-2.5-pro"
# FRAUD_NUM_CASES=50
# FRAUD_FRAUD_RATIO=0.4
```

> **E2B template:** `q1sg157kmhnqbfjth0ue` was built from a Dockerfile that downloads
> the full Kaggle `computingvictor/transactions-fraud-datasets` into `/data/` at build time.
> The CSVs are available inside every sandbox spawned from this template without any
> Kaggle credentials — only `E2B_API_KEY` is needed.

### 2. Install dependencies

```bash
uv sync
```

---

## Step-by-step: Build the database

The `create-db` command spins up a sandbox from the pre-built E2B template, pulls all
four dataset files from `/data/` inside the sandbox to your local `data/` directory,
then ingests them into SQLite. No Kaggle account needed.

```bash
# Pull CSVs from E2B template and build the SQLite DB (100k rows, stratified)
uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-db

# Smaller sample (faster for development)
uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-db --sample-size 20000

# If CSVs already downloaded locally, skip the E2B pull
uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-db --skip-download
```

The database is written to `implementations/fraud_analytics/data/fraud_transactions.db`.

---

## Step-by-step: Generate case files

```bash
# Generate 50 cases (40% fraud, 60% legit) — default settings
uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-cases

# Custom: 100 cases, 50% fraud, 14-day window
uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-cases \
  --num-cases 100 \
  --fraud-ratio 0.5 \
  --window-days 14
```

Output: `implementations/fraud_analytics/data/fraud_cases.jsonl`

---

## Run the Demo UI

```bash
uv run --env-file .env gradio implementations/fraud_analytics/gradio_app.py
```

Opens a Gradio interface. Select a Case ID from the dropdown and click **Investigate**.
The agent runs, calls SQL, optionally runs Python in E2B, checks accuracy, and returns
its verdict alongside the ground-truth label.

---

## Run the ADK Web UI (interactive inspection)

```bash
uv run adk web --port 8000 --reload --reload_agents implementations/
```

Navigate to http://localhost:8000. The `root_agent` from `agent.py` is auto-discovered.

---

## Run Evaluation

```bash
# Full evaluation on all cases
uv run --env-file .env python implementations/fraud_analytics/evaluate.py \
  --dataset-path implementations/fraud_analytics/data/fraud_cases.jsonl \
  --dataset-name fraud-analytics-eval

# Quick run: 10 cases only
uv run --env-file .env python implementations/fraud_analytics/evaluate.py \
  --dataset-name fraud-analytics-eval \
  --limit 10

# With custom concurrency and timeouts
uv run --env-file .env python implementations/fraud_analytics/evaluate.py \
  --dataset-name fraud-analytics-eval \
  --max-concurrent-cases 3 \
  --agent-timeout 180 \
  --llm-judge-timeout 60
```

### Evaluation output

The evaluation produces:

**Per-item metrics table** (printed to console):
```
Case ID      GT     Pred   Det    LLM    Correct
CASE_0001   FRAUD  FRAUD  0.900  4.25     ✅
CASE_0002   LEGIT  LEGIT  0.800  3.75     ✅
CASE_0003   FRAUD  LEGIT  0.200  2.00     ❌
```

**Run-level aggregate metrics**:
```
is_fraud classification (n=50)
  Precision : 0.8750
  Recall    : 0.8235
  F1 Score  : 0.8485
  Accuracy  : 0.8800

fraud_pattern macro-F1 : 0.6120
```

**All results uploaded to LangFuse** for dashboard visualisation.

---

## Evaluation Levels

### Item-level — Deterministic grader
| Metric | Weight | Description |
|--------|--------|-------------|
| `is_fraud_correct` | 40% | Verdict matches ground-truth label |
| `pattern_correct` | 20% | Fraud pattern matches ground truth (soft) |
| `seed_flagged` | 20% | Seed transaction appears in flagged IDs |
| `has_explanation` | 10% | Explanation is non-empty (>50 chars) |
| `calibration_ok` | 10% | Confidence consistent with correctness |

### Item-level — LLM-as-judge
Scores the `explanation` field on four dimensions (1–5 each):
- **Evidence grounding** — cites specific SQL results
- **Logical coherence** — reasoning follows from evidence
- **Pattern identification** — pattern named and justified
- **Confidence calibration** — confidence matches evidence strength

Rubric: `rubrics/explanation_quality.md`

### Trace-level — Deterministic
- SQL safety: all calls use SELECT only
- `check_accuracy` called at least once
- E2B executions succeeded (no error in output)
- No redundant SQL queries

### Run-level — Aggregate
- Precision / Recall / F1 / Accuracy for `is_fraud`
- Confusion matrix
- Macro-F1 for `fraud_pattern` classification
- All uploaded to LangFuse experiment

---

## File Structure

```
implementations/fraud_analytics/
├── env_vars.py              # pydantic-settings config
├── models.py                # FraudAnalysisOutput, CaseRecord, AccuracyResult
├── agent.py                 # Google ADK agent + 4 tools + runner helper
├── gradio_app.py            # Demo UI
├── evaluate.py              # 3-level evaluation pipeline
├── rubrics/
│   └── explanation_quality.md   # LLM-judge rubric
└── data/
    ├── schema.ddl           # SQLite schema (4 tables)
    ├── cli.py               # create-db and create-cases commands
    ├── fraud_transactions.db       # (generated — not committed)
    ├── fraud_cases.jsonl           # (generated — not committed)
    ├── fraud_cases_with_output.jsonl  # (generated — not committed)
    ├── transactions_data.csv       # (downloaded — not committed)
    ├── cards_data.csv              # (downloaded — not committed)
    ├── users_data.csv              # (downloaded — not committed)
    └── mcc_codes.json              # (downloaded — not committed)
```

---

## Dataset

**Source:** [computingvictor/transactions-fraud-datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)
on Kaggle. A synthetic financial transactions dataset for fraud detection and analytics.

Key tables after ingestion:
- `transactions` — core table with `is_fraud` ground-truth label
- `cards` — card metadata (brand, type, chip, credit limit)
- `users` — customer demographics and credit info
- `mcc_codes` — merchant category code lookup

---

## Safety Notes

The SQL tool enforces read-only access at three layers:
1. **Statement-level check** — only SELECT/WITH/EXPLAIN allowed
2. **Keyword scan** — blocks INSERT, UPDATE, DELETE, DROP, CREATE, etc.
3. **SQLite URI mode** — database opened as `file:...?mode=ro`
4. **SQLite authorizer** — write operations denied at the C-level authorizer

The E2B sandbox provides full isolation — agent-generated Python code runs in a
throwaway cloud VM with no access to local files or credentials.
