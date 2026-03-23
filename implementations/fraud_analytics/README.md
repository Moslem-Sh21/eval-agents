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
  └─ Downloads Kaggle dataset from E2B template → SQLite DB → fraud_cases.jsonl

Agent (agent.py)
  ├─ get_schema()        — inspect DB structure before querying
  ├─ execute_sql()       — read-only SQL (AST-level + URI mode + authorizer)
  ├─ check_accuracy()    — ground-truth label lookup by transaction ID
  └─ run_python()        — E2B secure Python sandbox with dataset at /data/

Demo UI (gradio_app.py)
  └─ Case selector → agent verdict + LangFuse score logging (Arrow 2)

Evaluation (evaluate.py)
  ├─ Arrow 3: agent output flows DIRECTLY into evaluators via RunResult
  ├─ Item-level:  deterministic grader (5 checks) + LLM-as-judge (4 dimensions)
  ├─ Tool-use:    SQL safety · E2B exec · check_accuracy called
  ├─ Run-level:   precision / recall / F1 · confusion matrix
  └─ Arrow 4: all scores uploaded to LangFuse
```

### How the Four Blocks Connect

```
┌─────────────────────────────────────────────────────────┐
│              Block 1: Fraud Analytics System            │
│  fraud_transactions.db  ←── data/cli.py                │
│  fraud_cases.jsonl      ←── data/cli.py                │
│  agent.py (4 tools)     ←── models.py (RunResult)      │
└───────────────────┬─────────────────────┬───────────────┘
                    │ Arrow 1             │ Arrow 3
          ┌─────────▼─────────┐  ┌───────▼──────────────┐
          │  Block 2: Demo UI  │  │ Block 3: Offline Eval │
          │  gradio_app.py     │  │ evaluate.py           │
          │  Arrow 2 ↓         │  │ Arrow 4 ↓             │
          └─────────┬──────────┘  └───────┬───────────────┘
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  LangFuse           │
                    │  Trace + Metric     │
                    │  Store              │
                    └─────────────────────┘
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
# FRAUD_EVALUATOR_MODEL="gemini-2.5-flash"
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
dataset files from `/data/` inside the sandbox to your local machine, joins
`train_fraud_labels.json` to add the `is_fraud` column (not present in the raw CSVs),
then ingests everything into SQLite.

```bash
# Pull CSVs from E2B template and build the SQLite DB (100k rows, stratified)
uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-db

# Smaller sample (faster for development)
uv run --env-file .env python -m implementations.fraud_analytics.data.cli create-db --sample-size 20000
```

The database is written to `implementations/fraud_analytics/data/fraud_transactions.db`.

> **Note on the dataset:** The raw `transactions_data.csv` does not contain an `is_fraud`
> column. Labels come from a separate `train_fraud_labels.json` file
> (`{"target": {"tx_id": "Yes"/"No"}}`). `create-db` joins these automatically.
> The resulting DB has ~0.09% fraud rate — realistic for production data.

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

Each case is a `CaseRecord` containing:
- `seed_transaction_id` — the transaction to investigate
- `client_id`, `card_id` — who and which card
- `window_start`, `window_end` — investigation time window
- `trigger_label` — what triggered the alert
- `ground_truth_is_fraud`, `ground_truth_pattern` — hidden from the agent, used for evaluation

---

## Run the Demo UI

```bash
uv run --env-file .env python -m implementations.fraud_analytics.gradio_app
```

Opens a Gradio interface at `http://localhost:7860`. Select a Case ID from the dropdown
and click **Investigate**. The agent runs, calls SQL, optionally runs Python in E2B,
checks accuracy, and returns its verdict. Results are automatically logged to LangFuse
(Arrow 2) with 5 scores per investigation: `ui_accuracy`, `confidence_score`,
`check_accuracy_called`, `sql_safe`, `e2b_called`.

---

## Run the ADK Web UI (interactive inspection)

```bash
uv run adk web --port 8000 --reload --reload_agents implementations/
```

Navigate to `http://localhost:8000`. The `root_agent` from `agent.py` is auto-discovered.

---

## Run Evaluation

Always run as a module (not as a direct file path):

```bash
# Full evaluation on all cases
uv run --env-file .env python -m implementations.fraud_analytics.evaluate \
  --dataset-name fraud-analytics-eval \
  --max-concurrent-cases 3

# Quick run: 5 cases only
uv run --env-file .env python -m implementations.fraud_analytics.evaluate \
  --dataset-name fraud-analytics-eval \
  --limit 5 \
  --max-concurrent-cases 3

# With custom timeouts
uv run --env-file .env python -m implementations.fraud_analytics.evaluate \
  --dataset-name fraud-analytics-eval \
  --max-concurrent-cases 3 \
  --agent-timeout 180 \
  --llm-judge-timeout 60
```

> **Concurrency:** Use `--max-concurrent-cases 3` to avoid Gemini API rate limits.
> Higher values speed up the run but risk throttling errors.

### Evaluation output

**Per-item metrics table** (printed to console):
```
Case ID      GT     Pred   Det    Tool   LLM    Acc?   Tools Called
CASE_0001   LEGIT  LEGIT  1.000  0.900  5.00    ✅     get_schema,execute_sql,...,check_accuracy
CASE_0013   FRAUD  FRAUD  1.000  0.900  2.25    ✅     get_schema,execute_sql,...,check_accuracy
CASE_0031   LEGIT  LEGIT  1.000  1.000  4.75    ✅     get_schema,execute_sql,run_python,...
```

**Tool-use summary:**
```
check_accuracy called : 50/50
SQL safety compliant  : 50/50
E2B (run_python) used : 5/50
```

**Run-level aggregate metrics:**
```
is_fraud classification (n=50)
  Precision : 1.0000
  Recall    : 1.0000
  F1 Score  : 1.0000
  Accuracy  : 1.0000

fraud_pattern macro-F1 : 0.6318
```

**Full results JSON** saved to:
```
implementations/fraud_analytics/data/eval_results_<run_name>.json
```

This file contains all three evaluator outputs per case including the LLM judge's
`brief_critique` — useful for understanding why specific cases scored low.

**All scores uploaded to LangFuse** under the run name `fraud_eval_<timestamp>`.

---

## Evaluation Levels

### 1. Item-level — Deterministic Grader (`deterministic_item_score`)

Five hard checks comparing agent output against ground truth. No LLM involved.

| Metric | Weight | Description |
|--------|--------|-------------|
| `is_fraud_correct` | 40% | Verdict matches ground-truth label |
| `pattern_correct` | 20% | Fraud pattern matches ground truth (soft — unknown patterns pass) |
| `seed_flagged` | 20% | Seed transaction ID appears in agent's flagged list |
| `has_explanation` | 10% | Explanation is non-empty (>50 chars) |
| `calibration_ok` | 10% | Confidence score is consistent with correctness |

### 2. Tool-use — Execution Grader (`tool_use_score`)

Evaluates **how** the agent worked, not just what answer it produced.
Reads directly from `RunResult` — no LangFuse polling needed (Arrow 3).

| Metric | Weight | Description |
|--------|--------|-------------|
| `check_accuracy_called` | 35% | Agent used the accuracy check tool |
| `sql_safe` | 35% | All SQL queries were read-only (SELECT only) |
| `calibration_ok` | 20% | Confidence consistent with correctness |
| `e2b_called` | 10% | Agent used Python sandbox when needed |

> Most cases score 0.9 here (check_accuracy + sql_safe + calibration = 0.90).
> Only cases that also used E2B (`run_python`) earn 1.0.

### 3. Item-level — LLM-as-Judge (`llm_judge_item_score`)

Scores the `explanation` field on four dimensions (1–5 each) using a second
Gemini model as the judge. Uses `google.genai` SDK with `response_mime_type="application/json"`
to ensure structured output.

| Dimension | Description |
|-----------|-------------|
| `evidence_grounding` | Cites specific SQL/Python results. Score 5 requires explicitly ruling out benign explanations |
| `logical_coherence` | Reasoning follows from evidence to verdict with no contradictions |
| `pattern_identification` | Pattern named AND mechanism explained with transaction-specific evidence |
| `confidence_calibration` | Confidence score matches strength of evidence |

Rubric: `rubrics/explanation_quality.md`

**Hard guardrails** (applied before scoring):
- Explanation contradicts `is_fraud` field → `logical_coherence = 1`, `overall_score = 1`
- Explanation names different pattern than `fraud_pattern` field → `pattern_identification = 1`
- Specific facts cited that couldn't come from queries (hallucination) → `evidence_grounding <= 2`
- Pattern field set but explanation has no mechanism description → `pattern_identification <= 2`

**Special cases:**
- `fraud_pattern = unknown`: don't penalize a coherent specific pattern if `is_fraud=true`
- E2B Python statistics count as valid evidence for `evidence_grounding`

### 4. Run-level — Aggregate Metrics (`compute_run_metrics`)

Computed after all cases complete using sklearn. Uploaded to LangFuse (Arrow 4).

| Metric | Description |
|--------|-------------|
| `is_fraud_precision` | Of all fraud predictions, how many were correct |
| `is_fraud_recall` | Of all actual frauds, how many were caught |
| `is_fraud_f1` | Harmonic mean of precision and recall |
| `is_fraud_accuracy` | Overall correct predictions / total cases |
| `fraud_pattern_macro_f1` | Macro-averaged F1 across all 8 pattern types |
| Confusion matrix | Full TP/FP/FN/TN breakdown |

---

## SQL Safety

The SQL tool enforces read-only access at four layers:

1. **AST-level check** — query is parsed as a syntax tree; non-SELECT statements rejected before execution
2. **Keyword scan** — blocks INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, ATTACH
3. **SQLite URI mode** — database opened as `file:...?mode=ro`
4. **Row limit** — results capped at 100 rows per query

This is why `SQL safety compliant: 50/50` — it is structurally impossible for the agent
to write to the database regardless of what SQL it generates.

---

## E2B Sandbox

The `run_python` tool spins up an isolated cloud VM (E2B) for each Python execution:

```
Request POST https://api.e2b.app/sandboxes  → sandbox created (~1 second)
Agent code executes inside the sandbox
The full Kaggle dataset is at /data/ inside every sandbox
DELETE https://api.e2b.app/sandboxes/{id}  → sandbox destroyed
```

Agent-generated Python code has no access to local files or credentials.
The sandbox is destroyed after every call — no state persists between `run_python` calls.

---

## VS Code Debugging

Create `.vscode/launch.json` to run and debug any pipeline step with a single click:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Create DB",
      "type": "python",
      "request": "launch",
      "module": "implementations.fraud_analytics.data.cli",
      "args": ["create-db"],
      "envFile": "${workspaceFolder}/.env",
      "cwd": "${workspaceFolder}",
      "justMyCode": false
    },
    {
      "name": "Create Cases",
      "type": "python",
      "request": "launch",
      "module": "implementations.fraud_analytics.data.cli",
      "args": ["create-cases"],
      "envFile": "${workspaceFolder}/.env",
      "cwd": "${workspaceFolder}",
      "justMyCode": false
    },
    {
      "name": "Gradio UI",
      "type": "python",
      "request": "launch",
      "module": "implementations.fraud_analytics.gradio_app",
      "envFile": "${workspaceFolder}/.env",
      "cwd": "${workspaceFolder}",
      "justMyCode": false
    },
    {
      "name": "Run Evaluation (50 cases)",
      "type": "python",
      "request": "launch",
      "module": "implementations.fraud_analytics.evaluate",
      "args": ["--dataset-name", "fraud-analytics-eval", "--max-concurrent-cases", "3"],
      "envFile": "${workspaceFolder}/.env",
      "cwd": "${workspaceFolder}",
      "justMyCode": false
    },
    {
      "name": "Run Evaluation (5 cases only)",
      "type": "python",
      "request": "launch",
      "module": "implementations.fraud_analytics.evaluate",
      "args": ["--dataset-name", "fraud-analytics-eval", "--max-concurrent-cases", "3", "--limit", "5"],
      "envFile": "${workspaceFolder}/.env",
      "cwd": "${workspaceFolder}",
      "justMyCode": false
    }
  ]
}
```

Set breakpoints in `agent.py` to inspect SQL queries as they execute, or in
`evaluate.py` after `llm_judge_item_score()` to read the full critique dict live.
Use the Debug Console to query live variables interactively while paused.

---

## File Structure

```
implementations/fraud_analytics/
├── env_vars.py              # pydantic-settings config (FraudAnalyticsSettings)
├── models.py                # FraudPattern, FraudAnalysisOutput, CaseRecord, RunResult
├── agent.py                 # Google ADK agent + 4 tools + run_case() orchestrator
├── gradio_app.py            # Demo UI + LangFuse score logging (Arrow 2)
├── evaluate.py              # 3-level evaluation pipeline (Arrows 3 & 4)
├── rubrics/
│   └── explanation_quality.md   # LLM-judge rubric with guardrails and special cases
└── data/
    ├── schema.ddl                     # SQLite schema (4 tables + indexes)
    ├── cli.py                         # create-db and create-cases CLI commands
    ├── fraud_transactions.db          # (generated — not committed)
    ├── fraud_cases.jsonl              # (generated — not committed)
    ├── eval_results_<run>.json        # (generated — not committed)
    ├── transactions_data.csv          # (downloaded from E2B — not committed)
    ├── cards_data.csv                 # (downloaded from E2B — not committed)
    ├── users_data.csv                 # (downloaded from E2B — not committed)
    ├── mcc_codes.json                 # (downloaded from E2B — not committed)
    └── train_fraud_labels.json        # (downloaded from E2B — not committed)
```

---

## Dataset

**Source:** [computingvictor/transactions-fraud-datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)
on Kaggle. A synthetic financial transactions dataset for fraud detection research.

Key tables after ingestion:
- `transactions` — core table with `is_fraud` ground-truth label (joined from labels file)
- `cards` — card metadata (brand, type, chip, credit limit)
- `users` — customer demographics and credit info
- `mcc_codes` — merchant category code lookup

Indexes are created on `client_id`, `card_id`, `date`, `is_fraud`, and `merchant_id`
for fast agent queries across large windows.

---

## Benchmark Results

Achieved on 50 cases (20 fraud, 30 legit) with `gemini-2.5-flash`:

| Metric | Score |
|--------|-------|
| `is_fraud` Precision | 1.0000 |
| `is_fraud` Recall | 1.0000 |
| `is_fraud` F1 | 1.0000 |
| `is_fraud` Accuracy | 1.0000 |
| `fraud_pattern` Macro-F1 | 0.6318 |
| `check_accuracy` called | 50/50 |
| SQL safety compliant | 50/50 |
| E2B used (spontaneous) | 5/50 |
| Avg LLM judge score | ~4.6/5.0 |

The agent achieves perfect fraud detection. Pattern classification (0.63 macro-F1) is
the remaining weak spot — boundaries between patterns like `card_not_present` and
`account_takeover` are genuinely ambiguous even for human analysts.

---

## Known LangFuse Compatibility Notes

This module uses **LangFuse v3**. The following v2 methods do not exist in v3:

| ❌ v2 (broken) | ✅ v3 (correct) |
|---|---|
| `langfuse.get_or_create_dataset()` | `langfuse.create_dataset()` |
| `langfuse.score()` | `langfuse.create_score()` |
| `langfuse.trace()` | Not needed — use `create_score` with a trace_id directly |

LangFuse logging is always non-fatal — a failure to log never blocks an investigation
or evaluation run.
