"""Fraud Analytics Evaluation Pipeline.

Runs a comprehensive three-level evaluation of the Fraud Analytics agent,
mirroring the structure used by the aml_investigation module:

  Item-level    — per-case deterministic grader + LLM-as-judge (explanation quality)
  Trace-level   — SQL safety, E2B execution success, check_accuracy call verification
  Run-level     — precision / recall / F1 for is_fraud; macro-F1 for fraud_pattern;
                  confusion matrix — all uploaded to LangFuse

Usage
-----
    uv run --env-file .env python implementations/fraud_analytics/evaluate.py \\
        --dataset-path implementations/fraud_analytics/data/fraud_cases.jsonl \\
        --dataset-name fraud-analytics-eval

Options
-------
    --agent-timeout           Seconds per agent run (default: 300)
    --llm-judge-timeout       Seconds per LLM judge call (default: 120)
    --llm-judge-retries       Retry attempts for LLM judge (default: 3)
    --max-concurrent-cases    Concurrent cases in batch (default: 5)
    --max-concurrent-traces   Concurrent trace evaluations (default: 10)
    --max-trace-wait-time     Max seconds to wait for LangFuse trace data (default: 300)
    --run-name                Experiment name in LangFuse (default: auto-generated)
    --limit                   Limit number of cases to evaluate (default: all)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from langfuse import Langfuse
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from implementations.fraud_analytics.agent import run_case
from implementations.fraud_analytics.env_vars import settings
from implementations.fraud_analytics.models import (
    CaseRecord,
    FraudAnalysisOutput,
    FraudPattern,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# LangFuse client
# ---------------------------------------------------------------------------

def _get_langfuse() -> Langfuse:
    return Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
    )


# ---------------------------------------------------------------------------
# Item-level evaluators
# ---------------------------------------------------------------------------

def deterministic_item_score(case: CaseRecord, output: FraudAnalysisOutput) -> dict:
    """Score a single case deterministically against ground truth.

    Returns a dict with boolean/float scores and a human-readable summary.
    """
    is_fraud_correct = output.is_fraud == case.ground_truth_is_fraud
    pattern_correct = (
        output.fraud_pattern == case.ground_truth_pattern
        or case.ground_truth_pattern == FraudPattern.UNKNOWN  # soft match
    )

    # Flagged IDs: does the seed transaction appear when fraud is present?
    seed_flagged = (
        case.seed_transaction_id in output.flagged_transaction_ids
        if case.ground_truth_is_fraud else True  # not penalised for legit cases
    )

    # Explanation non-empty
    has_explanation = bool(output.explanation and len(output.explanation.strip()) > 50)

    # Confidence calibration: high confidence for correct verdicts, low for wrong
    calibration_ok = (
        (is_fraud_correct and output.confidence_score >= 0.5) or
        (not is_fraud_correct and output.confidence_score < 0.7)
    )

    composite = (
        float(is_fraud_correct) * 0.4 +
        float(pattern_correct) * 0.2 +
        float(seed_flagged) * 0.2 +
        float(has_explanation) * 0.1 +
        float(calibration_ok) * 0.1
    )

    return {
        "is_fraud_correct": is_fraud_correct,
        "pattern_correct": pattern_correct,
        "seed_flagged": seed_flagged,
        "has_explanation": has_explanation,
        "calibration_ok": calibration_ok,
        "composite_score": round(composite, 3),
    }


async def llm_judge_item_score(
    case: CaseRecord,
    output: FraudAnalysisOutput,
    langfuse: Langfuse,
    model: str,
    timeout: int,
    retries: int,
) -> dict:
    """Score explanation quality using LLM-as-a-judge with the rubric.

    Returns a dict with per-dimension scores, overall_score, and critique.
    """
    rubric_path = Path(__file__).parent / "rubrics" / "explanation_quality.md"
    rubric = rubric_path.read_text() if rubric_path.exists() else "(rubric not found)"

    prompt = f"""{rubric}

---

## Case to Evaluate

**is_fraud:** {output.is_fraud}
**fraud_pattern:** {output.fraud_pattern.value}
**flagged_transaction_ids:** {output.flagged_transaction_ids}
**confidence_score:** {output.confidence_score}
**explanation:**
{output.explanation}
"""

    import google.generativeai as genai  # type: ignore

    for attempt in range(retries):
        try:
            model_client = genai.GenerativeModel(model)
            response = model_client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            text = response.text.strip()
            # Extract JSON
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group(0))
                # Log to LangFuse as a score on this trace
                return scores
        except Exception as e:
            logger.warning("LLM judge attempt %d/%d failed for case %s: %s", attempt + 1, retries, case.case_id, e)
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

    return {
        "evidence_grounding": None,
        "logical_coherence": None,
        "pattern_identification": None,
        "confidence_calibration": None,
        "overall_score": None,
        "brief_critique": "LLM judge failed after all retries.",
    }


# ---------------------------------------------------------------------------
# Trace-level evaluators
# ---------------------------------------------------------------------------

async def trace_deterministic_score(
    case: CaseRecord,
    langfuse: Langfuse,
    max_wait: int,
) -> dict:
    """Fetch trace from LangFuse and check tool-use safety properties.

    Checks:
      - sql_safety: all SQL tool calls used SELECT only
      - check_accuracy_called: agent called check_accuracy at least once
      - e2b_exec_success: if run_python was called, it succeeded (no error in output)
      - no_redundant_queries: no identical SQL queries in a row
    """
    # Give LangFuse time to ingest the trace
    deadline = time.time() + max_wait
    trace = None

    while time.time() < deadline:
        try:
            traces = langfuse.fetch_traces(
                tags=[case.case_id],
                limit=1,
            )
            if traces.data:
                trace = traces.data[0]
                break
        except Exception:
            pass
        await asyncio.sleep(5)

    if trace is None:
        return {
            "sql_safety": None,
            "check_accuracy_called": None,
            "e2b_exec_success": None,
            "no_redundant_queries": None,
            "status": "trace_not_found",
        }

    # Extract tool calls from trace observations
    try:
        observations = langfuse.fetch_observations(trace_id=trace.id).data
    except Exception:
        observations = []

    tool_calls = [
        obs for obs in observations
        if getattr(obs, "type", "") in ("TOOL", "tool") or
        "tool" in getattr(obs, "name", "").lower()
    ]

    # SQL safety: check all execute_sql inputs are SELECT-only
    sql_calls = [t for t in tool_calls if "execute_sql" in getattr(t, "name", "")]
    sql_safe = all(
        getattr(t, "input", {}).get("query", "").strip().lower().startswith(("select", "with", "explain"))
        for t in sql_calls
    ) if sql_calls else True  # no SQL calls = safe

    # check_accuracy called
    acc_calls = [t for t in tool_calls if "check_accuracy" in getattr(t, "name", "")]
    accuracy_called = len(acc_calls) > 0

    # E2B success
    e2b_calls = [t for t in tool_calls if "run_python" in getattr(t, "name", "")]
    e2b_ok = all(
        "❌" not in str(getattr(t, "output", ""))
        for t in e2b_calls
    ) if e2b_calls else None  # None if not called

    # Redundant queries
    sql_queries = [
        getattr(t, "input", {}).get("query", "").strip()
        for t in sql_calls
    ]
    has_redundancy = len(sql_queries) != len(set(sql_queries)) if sql_queries else False

    return {
        "sql_safety": sql_safe,
        "check_accuracy_called": accuracy_called,
        "e2b_exec_success": e2b_ok,
        "no_redundant_queries": not has_redundancy,
        "status": "ok",
        "tool_call_count": len(tool_calls),
        "sql_call_count": len(sql_calls),
        "e2b_call_count": len(e2b_calls),
    }


# ---------------------------------------------------------------------------
# Run-level aggregate metrics
# ---------------------------------------------------------------------------

def compute_run_metrics(
    cases: list[CaseRecord],
    outputs: dict[str, FraudAnalysisOutput | None],
) -> dict:
    """Compute precision/recall/F1 for is_fraud and pattern classification."""
    y_true_fraud, y_pred_fraud = [], []
    y_true_pattern, y_pred_pattern = [], []

    for case in cases:
        output = outputs.get(case.case_id)
        if output is None:
            continue

        y_true_fraud.append(int(case.ground_truth_is_fraud))
        y_pred_fraud.append(int(output.is_fraud))

        y_true_pattern.append(case.ground_truth_pattern.value)
        y_pred_pattern.append(output.fraud_pattern.value)

    if not y_true_fraud:
        return {"error": "no valid predictions to evaluate"}

    precision = precision_score(y_true_fraud, y_pred_fraud, zero_division=0)
    recall = recall_score(y_true_fraud, y_pred_fraud, zero_division=0)
    f1 = f1_score(y_true_fraud, y_pred_fraud, zero_division=0)
    accuracy = sum(a == b for a, b in zip(y_true_fraud, y_pred_fraud)) / len(y_true_fraud)

    cm = confusion_matrix(y_true_fraud, y_pred_fraud).tolist()

    pattern_f1_macro = f1_score(
        y_true_pattern, y_pred_pattern,
        average="macro",
        zero_division=0,
    )
    pattern_report = classification_report(
        y_true_pattern, y_pred_pattern,
        zero_division=0,
        output_dict=True,
    )

    return {
        "is_fraud": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "confusion_matrix": cm,
            "n_evaluated": len(y_true_fraud),
        },
        "fraud_pattern": {
            "macro_f1": round(pattern_f1_macro, 4),
            "per_class": pattern_report,
        },
    }


# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------

async def evaluate_async(
    dataset_path: str,
    dataset_name: str,
    run_name: str,
    agent_timeout: int,
    llm_judge_timeout: int,
    llm_judge_retries: int,
    max_concurrent_cases: int,
    max_concurrent_traces: int,
    max_trace_wait_time: int,
    limit: Optional[int],
) -> None:
    """Main async evaluation loop."""
    langfuse = _get_langfuse()

    # Load cases
    cases_path = Path(dataset_path)
    if not cases_path.exists():
        logger.error("Dataset not found: %s", cases_path)
        return

    cases: list[CaseRecord] = []
    with open(cases_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cases.append(CaseRecord.model_validate_json(line))
                except Exception as e:
                    logger.warning("Skipping malformed case: %s", e)

    if limit:
        cases = cases[:limit]

    click.echo(f"\n📋 Loaded {len(cases)} cases from {cases_path.name}")
    click.echo(f"   Fraud: {sum(c.ground_truth_is_fraud for c in cases)} | "
               f"Legit: {sum(not c.ground_truth_is_fraud for c in cases)}")

    # Upload dataset to LangFuse
    click.echo(f"\n📤 Uploading dataset to LangFuse as '{dataset_name}'...")
    try:
        lf_dataset = langfuse.get_or_create_dataset(name=dataset_name)
        for case in cases:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input={
                    "case_id": case.case_id,
                    "seed_transaction_id": case.seed_transaction_id,
                    "client_id": case.client_id,
                    "card_id": case.card_id,
                    "window_start": case.window_start,
                    "window_end": case.window_end,
                    "trigger_label": case.trigger_label,
                },
                expected_output={
                    "is_fraud": case.ground_truth_is_fraud,
                    "pattern": case.ground_truth_pattern.value,
                },
                metadata={"case_id": case.case_id},
            )
        click.echo("✅ Dataset uploaded.")
    except Exception as e:
        logger.warning("LangFuse dataset upload failed: %s", e)

    # Run agent on all cases
    click.echo(f"\n🤖 Running agent (max {max_concurrent_cases} concurrent, timeout={agent_timeout}s)...")
    semaphore = asyncio.Semaphore(max_concurrent_cases)
    outputs: dict[str, FraudAnalysisOutput | None] = {}
    errors: dict[str, str] = {}

    async def _run_one(case: CaseRecord) -> None:
        async with semaphore:
            try:
                output = await asyncio.wait_for(run_case(case), timeout=agent_timeout)
                outputs[case.case_id] = output
                status = "✅" if output else "⚠ (no output)"
                click.echo(f"  {case.case_id}: {status}")
            except asyncio.TimeoutError:
                errors[case.case_id] = "timeout"
                outputs[case.case_id] = None
                click.echo(f"  {case.case_id}: ⏱ timeout")
            except Exception as e:
                errors[case.case_id] = str(e)
                outputs[case.case_id] = None
                click.echo(f"  {case.case_id}: ❌ {e}")

    await asyncio.gather(*[_run_one(c) for c in cases])
    successful = sum(1 for v in outputs.values() if v is not None)
    click.echo(f"\n  Agent runs complete: {successful}/{len(cases)} successful")

    # Item-level evaluation
    click.echo("\n📊 Running item-level evaluation...")
    item_results: list[dict] = []

    for case in cases:
        output = outputs.get(case.case_id)
        if output is None:
            item_results.append({"case_id": case.case_id, "skipped": True})
            continue

        det = deterministic_item_score(case, output)
        llm = await llm_judge_item_score(
            case, output, langfuse,
            model=settings.evaluator_model,
            timeout=llm_judge_timeout,
            retries=llm_judge_retries,
        )

        item_result = {
            "case_id": case.case_id,
            "ground_truth_is_fraud": case.ground_truth_is_fraud,
            "predicted_is_fraud": output.is_fraud,
            "deterministic": det,
            "llm_judge": llm,
        }
        item_results.append(item_result)

        # Upload scores to LangFuse
        try:
            langfuse.score(
                trace_id=case.case_id,
                name="deterministic_composite",
                value=det["composite_score"],
                comment=f"is_fraud_correct={det['is_fraud_correct']}",
            )
            if llm.get("overall_score") is not None:
                langfuse.score(
                    trace_id=case.case_id,
                    name="explanation_quality",
                    value=llm["overall_score"],
                    comment=llm.get("brief_critique", ""),
                )
        except Exception as e:
            logger.debug("LangFuse score upload failed for %s: %s", case.case_id, e)

    # Print item-level table
    click.echo("\n" + "=" * 80)
    click.echo(f"{'Case ID':<12} {'GT':^6} {'Pred':^6} {'Det':^6} {'LLM':^6} {'Correct':^8}")
    click.echo("-" * 80)
    for r in item_results:
        if r.get("skipped"):
            click.echo(f"{r['case_id']:<12} {'—':^6} {'—':^6} {'—':^6} {'—':^6} {'SKIP':^8}")
            continue
        gt = "FRAUD" if r["ground_truth_is_fraud"] else "LEGIT"
        pred = "FRAUD" if r["predicted_is_fraud"] else "LEGIT"
        det_score = r["deterministic"]["composite_score"]
        llm_score = r["llm_judge"].get("overall_score") or "—"
        correct = "✅" if r["deterministic"]["is_fraud_correct"] else "❌"
        llm_str = f"{llm_score:.2f}" if isinstance(llm_score, float) else str(llm_score)
        click.echo(f"{r['case_id']:<12} {gt:^6} {pred:^6} {det_score:^6.3f} {llm_str:^6} {correct:^8}")
    click.echo("=" * 80)

    # Trace-level evaluation
    click.echo(f"\n🔍 Running trace-level evaluation (max {max_concurrent_traces} concurrent)...")
    trace_semaphore = asyncio.Semaphore(max_concurrent_traces)
    trace_results: list[dict] = []

    async def _eval_trace(case: CaseRecord) -> None:
        async with trace_semaphore:
            result = await trace_deterministic_score(case, langfuse, max_trace_wait_time)
            trace_results.append({"case_id": case.case_id, **result})

    await asyncio.gather(*[_eval_trace(c) for c in cases if outputs.get(c.case_id) is not None])

    successful_traces = sum(1 for t in trace_results if t.get("status") == "ok")
    acc_called = sum(1 for t in trace_results if t.get("check_accuracy_called") is True)
    sql_safe = sum(1 for t in trace_results if t.get("sql_safety") is True)

    click.echo(f"\n  Trace evaluation: {successful_traces}/{len(trace_results)} traces found")
    click.echo(f"  check_accuracy called: {acc_called}/{len(trace_results)}")
    click.echo(f"  SQL safety compliant: {sql_safe}/{len(trace_results)}")

    # Run-level metrics
    click.echo("\n📈 Computing run-level aggregate metrics...")
    run_metrics = compute_run_metrics(cases, outputs)

    click.echo("\n" + "=" * 80)
    click.echo("RUN-LEVEL METRICS")
    click.echo("=" * 80)
    if "error" in run_metrics:
        click.echo(f"❌ {run_metrics['error']}")
    else:
        fm = run_metrics["is_fraud"]
        click.echo(f"\n  is_fraud classification (n={fm['n_evaluated']})")
        click.echo(f"    Precision : {fm['precision']:.4f}")
        click.echo(f"    Recall    : {fm['recall']:.4f}")
        click.echo(f"    F1 Score  : {fm['f1']:.4f}")
        click.echo(f"    Accuracy  : {fm['accuracy']:.4f}")
        click.echo(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        cm = fm["confusion_matrix"]
        click.echo(f"             Pred:LEGIT  Pred:FRAUD")
        if len(cm) >= 2:
            click.echo(f"  True:LEGIT  {cm[0][0]:>9}  {cm[0][1]:>9}")
            click.echo(f"  True:FRAUD  {cm[1][0]:>9}  {cm[1][1]:>9}")

        pm = run_metrics["fraud_pattern"]
        click.echo(f"\n  fraud_pattern macro-F1 : {pm['macro_f1']:.4f}")

    click.echo("=" * 80)

    # Upload run-level metrics to LangFuse
    try:
        langfuse.score(
            trace_id=f"{run_name}_aggregate",
            name="is_fraud_f1",
            value=run_metrics.get("is_fraud", {}).get("f1", 0.0),
        )
        langfuse.score(
            trace_id=f"{run_name}_aggregate",
            name="is_fraud_precision",
            value=run_metrics.get("is_fraud", {}).get("precision", 0.0),
        )
        langfuse.score(
            trace_id=f"{run_name}_aggregate",
            name="is_fraud_recall",
            value=run_metrics.get("is_fraud", {}).get("recall", 0.0),
        )
        langfuse.score(
            trace_id=f"{run_name}_aggregate",
            name="pattern_macro_f1",
            value=run_metrics.get("fraud_pattern", {}).get("macro_f1", 0.0),
        )
        click.echo("\n✅ Aggregate metrics uploaded to LangFuse.")
    except Exception as e:
        logger.warning("LangFuse aggregate upload failed: %s", e)

    langfuse.flush()
    click.echo(f"\n✅ Evaluation complete. Run name: '{run_name}'")
    click.echo("   View results at https://us.cloud.langfuse.com")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--dataset-path", default=str(settings.cases_path), show_default=True)
@click.option("--dataset-name", default="fraud-analytics-eval", show_default=True)
@click.option("--run-name", default=None, help="Experiment run name (auto-generated if not set).")
@click.option("--agent-timeout", default=settings.agent_timeout, show_default=True)
@click.option("--llm-judge-timeout", default=120, show_default=True)
@click.option("--llm-judge-retries", default=3, show_default=True)
@click.option("--max-concurrent-cases", default=settings.max_concurrent_cases, show_default=True)
@click.option("--max-concurrent-traces", default=10, show_default=True)
@click.option("--max-trace-wait-time", default=300, show_default=True)
@click.option("--limit", default=None, type=int, help="Limit number of cases to evaluate.")
def main(
    dataset_path: str,
    dataset_name: str,
    run_name: Optional[str],
    agent_timeout: int,
    llm_judge_timeout: int,
    llm_judge_retries: int,
    max_concurrent_cases: int,
    max_concurrent_traces: int,
    max_trace_wait_time: int,
    limit: Optional[int],
) -> None:
    """Run the Fraud Analytics evaluation pipeline."""
    if run_name is None:
        run_name = f"fraud_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    asyncio.run(evaluate_async(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        run_name=run_name,
        agent_timeout=agent_timeout,
        llm_judge_timeout=llm_judge_timeout,
        llm_judge_retries=llm_judge_retries,
        max_concurrent_cases=max_concurrent_cases,
        max_concurrent_traces=max_concurrent_traces,
        max_trace_wait_time=max_trace_wait_time,
        limit=limit,
    ))


if __name__ == "__main__":
    main()
