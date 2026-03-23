"""Fraud Analytics Evaluation Pipeline.

Runs a comprehensive three-level evaluation of the Fraud Analytics agent,
following the architecture diagram exactly:

  Arrow 3: Agent output flows DIRECTLY into the Offline Evaluation Script.
           No LangFuse polling. evaluate.py calls run_case() and gets back
           a RunResult containing both the structured output and tool-call
           metadata for immediate evaluation.

  Arrow 4: Evaluation scores are uploaded to LangFuse (Trace + Metric Store).

Evaluation levels:
  Item-level    — per-case deterministic grader + LLM-as-judge (explanation quality)
  Tool-use      — SQL safety, check_accuracy called, E2B success (from RunResult directly)
  Run-level     — precision / recall / F1 for is_fraud; macro-F1 for fraud_pattern

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
    --run-name                Experiment name in LangFuse (default: auto-generated)
    --limit                   Limit number of cases to evaluate (default: all)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
    RunResult,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# LangFuse client (Arrow 4: Eval → LangFuse)
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

    Returns a dict with boolean/float scores and a composite score.
    """
    is_fraud_correct = output.is_fraud == case.ground_truth_is_fraud
    pattern_correct = (
        output.fraud_pattern == case.ground_truth_pattern
        or case.ground_truth_pattern == FraudPattern.UNKNOWN
    )
    seed_flagged = (
        case.seed_transaction_id in output.flagged_transaction_ids
        if case.ground_truth_is_fraud else True
    )
    has_explanation = bool(output.explanation and len(output.explanation.strip()) > 50)
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
    model: str,
    retries: int,
) -> dict:
    """Score explanation quality using LLM-as-a-judge with the rubric.

    Returns per-dimension scores (1-5) and an overall_score.
    """
    rubric_path = Path(__file__).parent / "rubrics" / "explanation_quality.md"
    rubric = rubric_path.read_text() if rubric_path.exists() else "(rubric not found)"

    # Keep prompt concise — prepend explicit JSON-only instruction so model
    # doesn't write a long preamble before the JSON object (causes MAX_TOKENS).
    prompt = f"""You are an evaluation judge. Score the fraud analysis explanation below using the rubric.
Respond ONLY with a valid JSON object — no preamble, no markdown, no extra text.

{rubric}

## Case
is_fraud: {output.is_fraud}
fraud_pattern: {output.fraud_pattern.value}
confidence_score: {output.confidence_score}
explanation (truncated to 500 chars): {output.explanation[:500] if output.explanation else ""}

Return exactly:
{{"evidence_grounding": <1-5>, "logical_coherence": <1-5>, "pattern_identification": <1-5>, "confidence_calibration": <1-5>, "overall_score": <1.0-5.0>, "brief_critique": "<max 10 words>"}}"""

    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore

    client = genai.Client()

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            # Safely extract text — finish_reason=MAX_TOKENS truncates parts
            text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = "".join(
                        p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
                    ).strip()

            if not text:
                finish = (
                    response.candidates[0].finish_reason
                    if response.candidates else "unknown"
                )
                raise ValueError(f"Empty response from LLM judge (finish_reason={finish})")

            # Try direct parse first (response_mime_type=json skips wrapping)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Fallback: regex for complete JSON object
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            # Last resort: if truncated mid-string, close the JSON manually
            if text.startswith("{") and not text.endswith("}"):
                try:
                    return json.loads(text + '"}'  + "}")
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"No JSON found in LLM judge response: {text[:200]}")
        except Exception as e:
            logger.warning(
                "LLM judge attempt %d/%d failed for case %s: %s",
                attempt + 1, retries, case.case_id, e,
            )
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
# Tool-use evaluator (replaces LangFuse trace polling — Arrow 3)
# ---------------------------------------------------------------------------

def tool_use_score(case: CaseRecord, run_result: RunResult) -> dict:
    """Evaluate tool-use quality DIRECTLY from RunResult (Arrow 3).

    This is the key fix: instead of waiting for LangFuse traces and polling
    the API, we evaluate the agent's tool-use behaviour directly from the
    RunResult returned by run_case(). RunResult carries:
      - tools_called: list of tool names from ADK events
      - sql_queries: list of SQL strings executed
      - e2b_called: bool
      - final_text: full agent response for fallback scanning

    Checks:
      - check_accuracy_called: agent used the accuracy check tool
      - sql_safe: no write keywords in any SQL query
      - e2b_called: agent used Python analysis when useful
      - calibration_ok: confidence is consistent with correctness
    """
    output = run_result.output
    if output is None:
        return {
            "check_accuracy_called": False,
            "sql_safe": True,
            "e2b_called": False,
            "calibration_ok": False,
            "composite_score": 0.0,
            "status": "no_output",
        }

    # check_accuracy called (from tool events OR final text fallback)
    check_accuracy_called = run_result.check_accuracy_called

    # SQL safety from captured queries + final text fallback
    sql_safe = run_result.sql_safe

    # E2B usage
    e2b_called = run_result.e2b_called

    # Confidence calibration
    is_correct = output.is_fraud == case.ground_truth_is_fraud
    calibration_ok = (
        (is_correct and output.confidence_score >= 0.5) or
        (not is_correct and output.confidence_score < 0.7)
    )

    composite = (
        float(check_accuracy_called) * 0.35 +
        float(sql_safe) * 0.35 +
        float(calibration_ok) * 0.20 +
        float(e2b_called) * 0.10
    )

    return {
        "check_accuracy_called": check_accuracy_called,
        "sql_safe": sql_safe,
        "e2b_called": e2b_called,
        "calibration_ok": calibration_ok,
        "tools_called": run_result.tools_called,
        "sql_query_count": len(run_result.sql_queries),
        "composite_score": round(composite, 3),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Run-level aggregate metrics
# ---------------------------------------------------------------------------

def compute_run_metrics(
    cases: list[CaseRecord],
    run_results: dict[str, RunResult],
) -> dict:
    """Compute precision/recall/F1 for is_fraud and pattern classification."""
    y_true_fraud, y_pred_fraud = [], []
    y_true_pattern, y_pred_pattern = [], []

    for case in cases:
        rr = run_results.get(case.case_id)
        if rr is None or rr.output is None:
            continue

        y_true_fraud.append(int(case.ground_truth_is_fraud))
        y_pred_fraud.append(int(rr.output.is_fraud))
        y_true_pattern.append(case.ground_truth_pattern.value)
        y_pred_pattern.append(rr.output.fraud_pattern.value)

    if not y_true_fraud:
        return {"error": "no valid predictions to evaluate"}

    precision = precision_score(y_true_fraud, y_pred_fraud, zero_division=0)
    recall = recall_score(y_true_fraud, y_pred_fraud, zero_division=0)
    f1 = f1_score(y_true_fraud, y_pred_fraud, zero_division=0)
    accuracy = sum(a == b for a, b in zip(y_true_fraud, y_pred_fraud)) / len(y_true_fraud)
    cm = confusion_matrix(y_true_fraud, y_pred_fraud).tolist()
    pattern_f1 = f1_score(y_true_pattern, y_pred_pattern, average="macro", zero_division=0)
    pattern_report = classification_report(y_true_pattern, y_pred_pattern, zero_division=0, output_dict=True)

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
            "macro_f1": round(pattern_f1, 4),
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
    click.echo(
        f"   Fraud: {sum(c.ground_truth_is_fraud for c in cases)} | "
        f"Legit: {sum(not c.ground_truth_is_fraud for c in cases)}"
    )

    # Upload dataset to LangFuse
    click.echo(f"\n📤 Uploading dataset to LangFuse as '{dataset_name}'...")
    try:
        langfuse.create_dataset(name=dataset_name)
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

    # -----------------------------------------------------------------------
    # Run agent on all cases (Arrow 3: direct output → eval script)
    # -----------------------------------------------------------------------
    click.echo(f"\n🤖 Running agent (max {max_concurrent_cases} concurrent, timeout={agent_timeout}s)...")
    semaphore = asyncio.Semaphore(max_concurrent_cases)
    run_results: dict[str, RunResult] = {}

    async def _run_one(case: CaseRecord) -> None:
        async with semaphore:
            try:
                rr = await asyncio.wait_for(run_case(case), timeout=agent_timeout)
                run_results[case.case_id] = rr
                status = "✅" if rr.output else "⚠ (no output)"
                click.echo(f"  {case.case_id}: {status} | tools: {rr.tools_called}")
            except asyncio.TimeoutError:
                run_results[case.case_id] = RunResult(
                    case_id=case.case_id, error="timeout"
                )
                click.echo(f"  {case.case_id}: ⏱ timeout")
            except Exception as e:
                run_results[case.case_id] = RunResult(
                    case_id=case.case_id, error=str(e)
                )
                click.echo(f"  {case.case_id}: ❌ {e}")

    await asyncio.gather(*[_run_one(c) for c in cases])
    successful = sum(1 for rr in run_results.values() if rr.output is not None)
    click.echo(f"\n  Agent runs complete: {successful}/{len(cases)} successful")

    # -----------------------------------------------------------------------
    # Item-level evaluation
    # -----------------------------------------------------------------------
    click.echo("\n📊 Running item-level evaluation...")
    item_results: list[dict] = []

    for case in cases:
        rr = run_results.get(case.case_id)
        if rr is None or rr.output is None:
            item_results.append({"case_id": case.case_id, "skipped": True})
            continue

        # Deterministic grader
        det = deterministic_item_score(case, rr.output)

        # LLM-as-judge (Explanation Evaluator)
        llm = await llm_judge_item_score(
            case, rr.output,
            model=settings.evaluator_model,
            retries=llm_judge_retries,
        )

        # Tool-use evaluator (Arrow 3 — direct from RunResult)
        tool = tool_use_score(case, rr)

        item_result = {
            "case_id": case.case_id,
            "ground_truth_is_fraud": case.ground_truth_is_fraud,
            "predicted_is_fraud": rr.output.is_fraud,
            "deterministic": det,
            "llm_judge": llm,
            "tool_use": tool,
        }
        item_results.append(item_result)

        # Arrow 4: upload item scores to LangFuse
        try:
            lf_trace_id = f"{run_name}_{case.case_id}"
            langfuse.create_score(
                trace_id=lf_trace_id,
                name="deterministic_composite",
                value=det["composite_score"],
                comment=f"is_fraud_correct={det['is_fraud_correct']}",
            )
            langfuse.create_score(
                trace_id=lf_trace_id,
                name="tool_use_composite",
                value=tool["composite_score"],
                comment=f"check_accuracy={tool['check_accuracy_called']}, sql_safe={tool['sql_safe']}",
            )
            if llm.get("overall_score") is not None:
                langfuse.create_score(
                    trace_id=lf_trace_id,
                    name="explanation_quality",
                    value=llm["overall_score"],
                    comment=llm.get("brief_critique", ""),
                )
        except Exception as e:
            logger.debug("LangFuse score upload failed for %s: %s", case.case_id, e)

    # Print item-level table
    click.echo("\n" + "=" * 100)
    click.echo(f"{'Case ID':<12} {'GT':^6} {'Pred':^6} {'Det':^6} {'Tool':^6} {'LLM':^6} {'Acc?':^8} {'Tools Called'}")
    click.echo("-" * 100)
    for r in item_results:
        if r.get("skipped"):
            click.echo(f"{r['case_id']:<12} {'—':^6} {'—':^6} {'—':^6} {'—':^6} {'—':^6} {'SKIP':^8}")
            continue
        gt = "FRAUD" if r["ground_truth_is_fraud"] else "LEGIT"
        pred = "FRAUD" if r["predicted_is_fraud"] else "LEGIT"
        det_s = r["deterministic"]["composite_score"]
        tool_s = r["tool_use"]["composite_score"]
        llm_s = r["llm_judge"].get("overall_score") or "—"
        correct = "✅" if r["deterministic"]["is_fraud_correct"] else "❌"
        llm_str = f"{llm_s:.2f}" if isinstance(llm_s, float) else str(llm_s)
        tools = ",".join(r["tool_use"].get("tools_called", []))
        click.echo(
            f"{r['case_id']:<12} {gt:^6} {pred:^6} {det_s:^6.3f} {tool_s:^6.3f} {llm_str:^6} {correct:^8} {tools}"
        )
    click.echo("=" * 100)

    # Tool-use summary
    checked = sum(1 for r in item_results if not r.get("skipped") and r["tool_use"]["check_accuracy_called"])
    sql_safe = sum(1 for r in item_results if not r.get("skipped") and r["tool_use"]["sql_safe"])
    e2b_used = sum(1 for r in item_results if not r.get("skipped") and r["tool_use"]["e2b_called"])
    evaluated = sum(1 for r in item_results if not r.get("skipped"))
    click.echo(f"\n  Tool-use summary ({evaluated} evaluated):")
    click.echo(f"  check_accuracy called : {checked}/{evaluated}")
    click.echo(f"  SQL safety compliant  : {sql_safe}/{evaluated}")
    click.echo(f"  E2B (run_python) used : {e2b_used}/{evaluated}")

    # -----------------------------------------------------------------------
    # Run-level aggregate metrics (Arrow 4: upload to LangFuse)
    # -----------------------------------------------------------------------
    click.echo("\n📈 Computing run-level aggregate metrics...")
    run_metrics = compute_run_metrics(cases, run_results)

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

    # Arrow 4: upload aggregate metrics to LangFuse
    try:
        agg_id = f"{run_name}_aggregate"
        if "is_fraud" in run_metrics:
            langfuse.create_score(trace_id=agg_id, name="is_fraud_f1", value=run_metrics["is_fraud"]["f1"])
            langfuse.create_score(trace_id=agg_id, name="is_fraud_precision", value=run_metrics["is_fraud"]["precision"])
            langfuse.create_score(trace_id=agg_id, name="is_fraud_recall", value=run_metrics["is_fraud"]["recall"])
            langfuse.create_score(trace_id=agg_id, name="is_fraud_accuracy", value=run_metrics["is_fraud"]["accuracy"])
            langfuse.create_score(trace_id=agg_id, name="pattern_macro_f1", value=run_metrics["fraud_pattern"]["macro_f1"])
        click.echo("\n✅ Aggregate metrics uploaded to LangFuse.")
    except Exception as e:
        logger.warning("LangFuse aggregate upload failed: %s", e)

    # Save full results to JSON (includes LLM judge critiques per case)
    results_path = Path(dataset_path).parent / f"eval_results_{run_name}.json"
    with open(results_path, "w") as f:
        json.dump(item_results, f, indent=2, default=str)
    click.echo(f"\n💾 Full results saved to {results_path}")

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
@click.option("--limit", default=None, type=int, help="Limit number of cases to evaluate.")
def main(
    dataset_path: str,
    dataset_name: str,
    run_name: Optional[str],
    agent_timeout: int,
    llm_judge_timeout: int,
    llm_judge_retries: int,
    max_concurrent_cases: int,
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
        limit=limit,
    ))


if __name__ == "__main__":
    main()
