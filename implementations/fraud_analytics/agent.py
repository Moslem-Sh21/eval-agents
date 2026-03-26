"""Fraud Analytics Agent.

A Google ADK agent backed by Gemini that investigates fraud cases using four tools:
  - get_schema()       : inspect the DB schema before querying
  - execute_sql()      : run read-only SQL against the transactions database
  - check_accuracy()   : look up ground-truth fraud label for a transaction ID
  - run_python()       : execute Python analysis code in an E2B sandbox

The agent is instructed to follow a structured investigation workflow and
produce a FraudAnalysisOutput JSON block as its final response.

run_case() returns a RunResult that carries both the structured output AND
the raw agent text + tool-call metadata. This allows evaluate.py to evaluate
the agent's tool-use behaviour directly (Arrow 3 in the architecture diagram)
without needing to poll LangFuse for trace data.

The module exposes `root_agent` for ADK web UI discovery:
    uv run adk web --port 8000 --reload --reload_agents implementations/
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from google.adk.agents import Agent
from google.genai import types as genai_types

from implementations.fraud_analytics.env_vars import settings
from implementations.fraud_analytics.models import (
    AccuracyResult,
    CaseRecord,
    FraudAnalysisOutput,
    RunResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool: get_schema
# ---------------------------------------------------------------------------

def get_schema() -> str:
    """Return the database schema as a markdown-formatted string.

    The agent should call this tool FIRST before writing any SQL queries.
    It shows all tables, columns, and types so the agent can write correct queries.

    Returns:
        A markdown table listing all tables and their columns.
    """
    db_path = Path(settings.db.database)
    if not db_path.exists():
        return "❌ Database not found. Run `cli.py create-db` first."

    conn = sqlite3.connect(db_path)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()

        lines = ["## Database Schema\n"]
        for (table_name,) in tables:
            lines.append(f"### `{table_name}`\n")
            cols = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
            lines.append("| # | Column | Type | NotNull | PK |")
            lines.append("|---|--------|------|---------|-----|")
            for col in cols:
                lines.append(f"| {col[0]} | `{col[1]}` | {col[2]} | {bool(col[3])} | {bool(col[5])} |")
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
            lines.append(f"\n*{count:,} rows*\n")

        return "\n".join(lines)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool: execute_sql
# ---------------------------------------------------------------------------

_MAX_ROWS = 100
_ALLOWED_STATEMENT_STARTS = {"select", "with", "explain"}
_BLOCKED_KEYWORDS = {
    "insert", "update", "delete", "drop", "create", "alter",
    "truncate", "replace", "attach", "detach", "pragma",
}


def execute_sql(query: str) -> str:
    """Execute a read-only SQL query against the fraud transactions database.

    IMPORTANT RESTRICTIONS:
    - Only SELECT / WITH / EXPLAIN statements are allowed.
    - No INSERT, UPDATE, DELETE, DROP, CREATE, or any write operation.
    - Results are limited to 100 rows maximum.

    Args:
        query: A valid read-only SQL query string.

    Returns:
        Query results formatted as a markdown table, or an error message.
    """
    stripped = query.strip().lstrip("(").lower()
    first_word = stripped.split()[0] if stripped.split() else ""

    if first_word not in _ALLOWED_STATEMENT_STARTS:
        return (
            f"❌ SQL blocked: statement starts with '{first_word}'. "
            f"Only SELECT/WITH/EXPLAIN queries are permitted."
        )

    query_lower = query.lower()
    for kw in _BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", query_lower):
            return f"❌ SQL blocked: contains forbidden keyword '{kw}'."

    db_path = Path(settings.db.database)
    if not db_path.exists():
        return "❌ Database not found. Run `cli.py create-db` first."

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.set_authorizer(_read_only_authorizer)

    try:
        conn.execute("PRAGMA query_only = ON;")
        cursor = conn.execute(query)
        rows = cursor.fetchmany(_MAX_ROWS)
        col_names = [d[0] for d in cursor.description] if cursor.description else []

        if not rows:
            return "*(query returned 0 rows)*"

        lines = ["| " + " | ".join(col_names) + " |"]
        lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(v) if v is not None else "NULL" for v in row) + " |")

        suffix = ""
        if len(rows) == _MAX_ROWS:
            suffix = f"\n\n*Results limited to {_MAX_ROWS} rows. Refine your query to see more.*"

        return "\n".join(lines) + suffix

    except sqlite3.OperationalError as e:
        return f"❌ SQL error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"
    finally:
        conn.close()


def _read_only_authorizer(action_code: int, *args: Any) -> int:
    """SQLite authorizer that blocks all write operations."""
    import sqlite3 as _sqlite3
    _WRITE_ACTIONS = {
        _sqlite3.SQLITE_INSERT,
        _sqlite3.SQLITE_UPDATE,
        _sqlite3.SQLITE_DELETE,
        _sqlite3.SQLITE_CREATE_TABLE,
        _sqlite3.SQLITE_DROP_TABLE,
        _sqlite3.SQLITE_ALTER_TABLE,
        _sqlite3.SQLITE_ATTACH,
    }
    if action_code in _WRITE_ACTIONS:
        return _sqlite3.SQLITE_DENY
    return _sqlite3.SQLITE_OK


# ---------------------------------------------------------------------------
# Tool: check_accuracy
# ---------------------------------------------------------------------------

_case_index: dict[str, CaseRecord] | None = None


def _load_case_index() -> dict[str, CaseRecord]:
    """Load and cache case records keyed by seed_transaction_id."""
    global _case_index
    if _case_index is not None:
        return _case_index

    cases_path = Path(settings.cases_path)
    if not cases_path.exists():
        logger.warning("Cases file not found at %s — check_accuracy will always return not found.", cases_path)
        _case_index = {}
        return _case_index

    index: dict[str, CaseRecord] = {}
    with open(cases_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = CaseRecord.model_validate_json(line)
                index[case.seed_transaction_id] = case
            except Exception as e:
                logger.warning("Could not parse case line: %s — %s", line[:80], e)

    _case_index = index
    logger.info("Loaded %d cases into check_accuracy index.", len(index))
    return _case_index


def check_accuracy(transaction_id: str, predicted_is_fraud: bool) -> str:
    """Look up the ground-truth fraud label for a transaction and compare with your prediction.

    Use this tool AFTER you have completed your SQL investigation and formed
    a verdict. It reveals whether your prediction matches the dataset label,
    which you should factor into your explanation and confidence score.

    Args:
        transaction_id: The transaction ID you investigated (the seed transaction).
        predicted_is_fraud: Your current prediction — True if you believe it is fraud.

    Returns:
        A JSON string containing the ground truth label and whether your
        prediction matched. Incorporate this into your final verdict.
    """
    index = _load_case_index()

    if transaction_id not in index:
        db_path = Path(settings.db.database)
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "SELECT is_fraud FROM transactions WHERE id = ? LIMIT 1",
                    (transaction_id,),
                ).fetchone()
                if row is not None:
                    ground_truth = bool(row[0])
                    match = ground_truth == predicted_is_fraud
                    result = AccuracyResult(
                        transaction_id=transaction_id,
                        ground_truth_is_fraud=ground_truth,
                        agent_prediction=predicted_is_fraud,
                        match=match,
                        message=(
                            f"✅ Prediction CORRECT — transaction {transaction_id} is "
                            f"{'FRAUDULENT' if ground_truth else 'LEGITIMATE'} in the ground truth."
                            if match else
                            f"❌ Prediction INCORRECT — transaction {transaction_id} is "
                            f"{'FRAUDULENT' if ground_truth else 'LEGITIMATE'} in the ground truth, "
                            f"but you predicted {'fraud' if predicted_is_fraud else 'legitimate'}. "
                            f"Review your evidence and reconsider your confidence score."
                        ),
                    )
                    return result.model_dump_json(indent=2)
            finally:
                conn.close()

        return json.dumps({
            "error": f"Transaction ID '{transaction_id}' not found in case index or database.",
            "hint": "Ensure the transaction_id matches the seed_transaction_id for this case.",
        })

    case = index[transaction_id]
    ground_truth = case.ground_truth_is_fraud
    match = ground_truth == predicted_is_fraud

    result = AccuracyResult(
        transaction_id=transaction_id,
        ground_truth_is_fraud=ground_truth,
        agent_prediction=predicted_is_fraud,
        match=match,
        message=(
            f"✅ Prediction CORRECT — transaction {transaction_id} is "
            f"{'FRAUDULENT' if ground_truth else 'LEGITIMATE'} in the ground truth."
            if match else
            f"❌ Prediction INCORRECT — transaction {transaction_id} is "
            f"{'FRAUDULENT' if ground_truth else 'LEGITIMATE'} in the ground truth, "
            f"but you predicted {'fraud' if predicted_is_fraud else 'legitimate'}. "
            f"Review your evidence and reconsider your confidence score."
        ),
    )
    return result.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Tool: run_python
# ---------------------------------------------------------------------------

def run_python(code: str) -> str:
    """Execute Python code in a secure E2B cloud sandbox for deeper analysis.

    Use this tool when SQL alone is insufficient — for example:
    - Statistical analysis of a cardholder's transaction history
    - Computing velocity metrics (transactions per hour/day)
    - Visualising spending patterns as text/ASCII
    - Calculating deviation scores or z-scores

    IMPORTANT: The full Kaggle transactions-fraud dataset is pre-loaded inside
    the sandbox at these paths:
        /data/transactions_data.csv  — all transactions (including is_fraud label)
        /data/cards_data.csv         — card metadata
        /data/users_data.csv         — user/customer metadata
        /data/mcc_codes.json         — merchant category codes

    The sandbox has pandas, numpy, scipy, and matplotlib available.
    Use print() to return results. Do not write files — use print() for output.

    Example:
        import pandas as pd
        df = pd.read_csv('/data/transactions_data.csv')
        print(df[df['client_id'] == '12345']['amount'].describe())

    Args:
        code: Valid Python code to execute. Use print() to return results.

    Returns:
        stdout output from the executed code, or an error message.
    """
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return (
            "❌ e2b_code_interpreter is not installed. "
            "Add it to pyproject.toml: e2b-code-interpreter>=2.4.1"
        )

    api_key = settings.e2b_api_key or None
    template_id = settings.e2b_template_id

    sandbox = None
    try:
        sandbox = Sandbox.create(template=template_id, api_key=api_key)
        execution = sandbox.run_code(code)
        output_parts = []
        if execution.logs.stdout:
            output_parts.append("**stdout:**\n" + "\n".join(execution.logs.stdout))
        if execution.logs.stderr:
            output_parts.append("**stderr:**\n" + "\n".join(execution.logs.stderr))
        if execution.error:
            output_parts.append(f"**error:** {execution.error.name}: {execution.error.value}")
            if execution.error.traceback:
                output_parts.append("**traceback:**\n" + execution.error.traceback)
        if not output_parts:
            return "*(code executed with no output)*"
        return "\n\n".join(output_parts)
    except Exception as e:
        return f"❌ E2B sandbox error: {type(e).__name__}: {e}"
    finally:
        if sandbox is not None:
            sandbox.kill()


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """You are an expert fraud investigator. Your job is to investigate financial transactions
and determine whether they are fraudulent.

## Investigation Workflow

For each case you receive, follow these steps in order:

1. **Understand the schema** — Call `get_schema()` first to understand what tables and columns are available.

2. **Examine the seed transaction** — Query the `transactions` table for the seed_transaction_id provided.
   Note the amount, merchant, use_chip type, date, client_id, and card_id.

3. **Check the account history** — Query the last 30 days of transactions for the same client_id and card_id
   within the investigation window (window_start to window_end). Look for:
   - Unusual amounts relative to the account's history
   - Repeated merchant IDs or suspicious merchant names
   - Multiple declined transactions (errors column)
   - Transactions in different geographic locations in short time spans
   - Online transactions ("Online Transaction" in use_chip) followed by in-person ones

4. **Run deeper analysis if needed** — Use `run_python()` to compute velocity metrics,
   deviation scores, or statistical anomalies that SQL alone cannot easily express.
   If you found a clear fraud signal in steps 2–3, additional queries are optional.
   If the evidence is ambiguous, run more queries before concluding.

5. **Challenge your verdict** — Before writing anything, ask yourself:
   "What is the most innocent explanation for this transaction?"
   Consider these benign alternatives:
   - Could this be a one-time large purchase (holiday, travel, gift)?
   - Is the client's spending history simply variable by nature?
   - Could the merchant be legitimate but just unfamiliar to this client?
   - Could the velocity be explained by a single shopping trip or event?
   Use this to calibrate your confidence, not to override clear evidence:
   - If the evidence strongly points to fraud AND the benign explanation is
     inconsistent with your query results → classify as FRAUD with high confidence.
   - If the evidence is ambiguous and a benign explanation is plausible →
     lower your confidence score (0.5–0.7) but still classify based on the
     balance of evidence, not on whether you can rule out every alternative.
   - Only classify as LEGIT if the evidence does NOT support fraud, not merely
     because a benign explanation is theoretically possible.

6. **Write your explanation** — Write the explanation field NOW, based ONLY on the
   SQL and Python evidence you gathered in steps 1–5. Your explanation must be
   complete and self-contained at this point.
   IMPORTANT: Treat this as the FINAL version of your explanation. You will NOT
   update or change it after this step regardless of what you learn next.
   The explanation must include:
   - The exact amount of the seed transaction
   - The client's average transaction amount from your queries
   - The merchant name or ID
   - At least one direct comparison to the client's historical behavior
   - Your assessment of the most plausible innocent explanation and why the
     evidence does or does not support it

7. **Verify consistency** — Check that your planned JSON output is consistent with
   the explanation you just wrote:
   - Explanation concludes fraud? → is_fraud must be true
   - Explanation concludes legitimate? → is_fraud must be false
   - Named a pattern in explanation? → fraud_pattern must match exactly
   - Confidence score reflects the evidence strength you described?
   Fix ANY mismatch now. A contradictory output is treated as a complete failure.

8. **Call check_accuracy** — Call `check_accuracy(transaction_id, predicted_is_fraud)`
   with your verdict. This is a FINAL SEAL only — you have already written your
   explanation and it is locked. Do NOT go back and change your explanation or
   reasoning based on what check_accuracy returns. The result does not affect
   your explanation in any way.

9. **Produce your final output** — Return ONLY a JSON block matching this exact schema:
   ```json
   {
     "is_fraud": true,
     "fraud_pattern": "unusual_velocity",
     "flagged_transaction_ids": ["TXN_001", "TXN_002"],
     "confidence_score": 0.85,
     "explanation": "..."
   }
   ```
   Use the explanation you wrote in step 6 exactly as written.

## How to Choose the Fraud Pattern

Use this decision order — pick the FIRST one that matches the seed transaction:

1. Was the transaction online with no chip/PIN present? → `card_not_present`
2. Are there signs of a bad PIN, failed login, or access from a new location? → `account_takeover`
3. Is the merchant category completely new for this client with no prior history? → `identity_theft`
4. Are there many small transactions just under a round number threshold (e.g. $99, $499)? → `smurfing`
5. Is the merchant ID unknown or the MCC code suspicious/mismatched? → `merchant_fraud`
6. Are there more than 5 transactions within a 2-hour window? → `unusual_velocity`
7. Are transactions happening in geographically impossible locations within hours of each other? → `geo_anomaly`
8. Fraud is clearly present but none of the above patterns fit? → `unknown`
9. Transaction is legitimate → `none`

If multiple patterns seem to apply, pick the ONE that best explains the seed transaction specifically.
Valid values: `card_not_present`, `account_takeover`, `identity_theft`, `smurfing`,
`merchant_fraud`, `unusual_velocity`, `geo_anomaly`, `unknown`, `none`

## Confidence Score Guidelines

Use these thresholds — be honest about uncertainty:

- 0.9 – 1.0 : Multiple independent signals all point to fraud AND you explicitly
               ruled out all benign explanations with specific evidence from queries.
- 0.7 – 0.9 : Strong evidence but at least one signal could have an innocent explanation.
- 0.5 – 0.7 : Moderate — transaction is suspicious but not conclusive.
- 0.3 – 0.5 : Weak — unusual but no clear fraud signal. Classify as LEGIT.
- 0.0 – 0.3 : No meaningful fraud signals found.

NEVER assign confidence > 0.7 if:
- You only ran 1 SQL query
- You found only a single suspicious signal
- You could not rule out a benign explanation with specific data

## Minimum Evidence Requirements

Your explanation MUST reference ALL of the following if available:
- The exact amount of the seed transaction
- The client's average transaction amount (compute from query history if needed)
- The merchant name or ID
- The date and time of the seed transaction
- At least one direct comparison to the client's historical behavior

An explanation that does not cite specific numbers from your queries will score
poorly regardless of whether the verdict is correct. Give a reviewer enough
specific data to verify your reasoning independently.

## Important Rules
- Stay within the investigation window (window_start to window_end).
- Only use SELECT queries — no writes, no schema changes.
- Your explanation (written in step 6) must stand entirely on the SQL and Python
  evidence from your queries. It must never reference check_accuracy or its result.
- Vague claims without specific numbers are not acceptable. Always cite exact
  amounts, dates, merchant IDs, and transaction counts from your query results.
- Classify based on the balance of evidence. A theoretically possible innocent
  explanation does not override strong fraud signals in the data.
"""


root_agent = Agent(
    name="fraud_analytics_agent",
    model=settings.agent_model,
    description=(
        "A fraud investigation agent that queries a financial transactions database "
        "using read-only SQL, verifies ground-truth labels via check_accuracy, and "
        "optionally runs Python analysis in an E2B sandbox."
    ),
    instruction=_SYSTEM_INSTRUCTION,
    tools=[get_schema, execute_sql, check_accuracy, run_python],
)


# ---------------------------------------------------------------------------
# Runner — returns RunResult for direct evaluation (Arrow 3)
# ---------------------------------------------------------------------------

async def run_case(case: CaseRecord) -> RunResult:
    """Run the agent on a single CaseRecord and return a RunResult.

    RunResult carries both the parsed FraudAnalysisOutput AND the raw agent
    text plus tool-call metadata. This lets evaluate.py assess tool-use
    behaviour directly without polling LangFuse (implements Arrow 3).

    Returns a RunResult with output=None if the agent response cannot be parsed.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="fraud_analytics",
        user_id="batch_runner",
        session_id=case.case_id,
    )

    runner = Runner(
        agent=root_agent,
        app_name="fraud_analytics",
        session_service=session_service,
    )

    prompt = (
        f"Investigate the following fraud case:\n\n"
        f"**Case ID:** {case.case_id}\n"
        f"**Seed Transaction ID:** {case.seed_transaction_id}\n"
        f"**Client ID:** {case.client_id}\n"
        f"**Card ID:** {case.card_id}\n"
        f"**Investigation Window:** {case.window_start} to {case.window_end}\n"
        f"**Trigger:** {case.trigger_label}\n\n"
        f"Follow the investigation workflow. End with a JSON block containing your verdict."
    )

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt)],
    )

    final_text = ""
    tools_called: list[str] = []
    sql_queries: list[str] = []
    e2b_called = False

    async for event in runner.run_async(
        user_id="batch_runner",
        session_id=case.case_id,
        new_message=user_message,
    ):
        # Capture final response text
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text for p in event.content.parts if hasattr(p, "text"))

        # Capture tool calls from intermediate events
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                # ADK emits function_call parts for tool invocations
                if hasattr(part, "function_call") and part.function_call:
                    tool_name = part.function_call.name
                    tools_called.append(tool_name)
                    if tool_name == "execute_sql":
                        args = part.function_call.args or {}
                        query = args.get("query", "")
                        if query:
                            sql_queries.append(query)
                    if tool_name == "run_python":
                        e2b_called = True

    # Parse structured output from final text
    output: FraudAnalysisOutput | None = None
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", final_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{[^{}]*\"is_fraud\"[^{}]*\})", final_text, re.DOTALL)

    if json_match:
        try:
            raw_json = json_match.group(1)
            # Normalize fraud_pattern before Pydantic validation.
            # The agent occasionally produces close variants of valid enum values
            # (e.g. "identity_thievery" instead of "identity_theft"). We map
            # these to the nearest valid value so the case isn't skipped entirely.
            _PATTERN_ALIASES: dict[str, str] = {
                "identity_thievery": "identity_theft",
                "identity theft": "identity_theft",
                "card not present": "card_not_present",
                "account takeover": "account_takeover",
                "unusual velocity": "unusual_velocity",
                "geo anomaly": "geo_anomaly",
                "merchant fraud": "merchant_fraud",
                "card_not_present_fraud": "card_not_present",
                "velocity": "unusual_velocity",
                "geographic_anomaly": "geo_anomaly",
            }
            try:
                parsed = json.loads(raw_json)
                raw_pattern = str(parsed.get("fraud_pattern", "")).strip().lower()
                if raw_pattern in _PATTERN_ALIASES:
                    logger.warning(
                        "Normalizing fraud_pattern '%s' → '%s' for case %s",
                        raw_pattern, _PATTERN_ALIASES[raw_pattern], case.case_id,
                    )
                    parsed["fraud_pattern"] = _PATTERN_ALIASES[raw_pattern]
                    raw_json = json.dumps(parsed)
            except json.JSONDecodeError:
                pass  # let Pydantic handle the error below

            output = FraudAnalysisOutput.model_validate_json(raw_json)
        except Exception as e:
            logger.warning("Failed to parse agent output for %s: %s", case.case_id, e)

    if output is None:
        logger.warning("No valid JSON output found for case %s", case.case_id)

    return RunResult(
        case_id=case.case_id,
        output=output,
        final_text=final_text,
        tools_called=tools_called,
        sql_queries=sql_queries,
        e2b_called=e2b_called,
    )
