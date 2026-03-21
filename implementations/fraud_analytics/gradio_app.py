"""Fraud Analytics Demo UI.

Gradio app that accepts a Case ID from fraud_cases.jsonl, runs the agent,
and displays the verdict with accuracy metrics inline.

Arrow 2 in the architecture diagram (UI → LangFuse) is implemented here:
every time a user clicks Investigate, the query and agent response are logged
as a LangFuse trace so they appear in the Trace + Metric Store.

Run:
    uv run --env-file .env python -m implementations.fraud_analytics.gradio_app
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import gradio as gr
from langfuse import Langfuse

from implementations.fraud_analytics.agent import run_case
from implementations.fraud_analytics.env_vars import settings
from implementations.fraud_analytics.models import CaseRecord, RunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangFuse client (Arrow 2: UI → LangFuse)
# ---------------------------------------------------------------------------

def _get_langfuse() -> Langfuse:
    """Return a configured LangFuse client using env vars."""
    return Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
    )


# ---------------------------------------------------------------------------
# Load cases for the dropdown
# ---------------------------------------------------------------------------

def _load_cases() -> dict[str, CaseRecord]:
    """Return a dict of case_id -> CaseRecord from the JSONL file."""
    cases_path = Path(settings.cases_path)
    if not cases_path.exists():
        return {}
    cases = {}
    with open(cases_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = CaseRecord.model_validate_json(line)
                cases[case.case_id] = case
            except Exception:
                pass
    return cases


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _verdict_badge(is_fraud: bool) -> str:
    return "🔴 FRAUD" if is_fraud else "🟢 LEGITIMATE"


def _confidence_bar(score: float) -> str:
    filled = int(score * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score:.0%}"


def _match_badge(match: bool | None) -> str:
    if match is None:
        return "—"
    return "✅ Correct" if match else "❌ Incorrect"


# ---------------------------------------------------------------------------
# Core investigation function
# ---------------------------------------------------------------------------

def run_investigation(case_id: str) -> tuple[str, str, str, str, str, str, str, str]:
    """Called by Gradio on button click.

    Implements Arrow 2: logs query + result to LangFuse on every run.
    Returns (trigger, ground_truth, verdict, pattern, confidence, flagged, explanation, accuracy).
    """
    cases = _load_cases()

    if not cases:
        msg = (
            "⚠ No cases found. Run:\n"
            "  python -m implementations.fraud_analytics.data.cli create-db\n"
            "  python -m implementations.fraud_analytics.data.cli create-cases"
        )
        return (msg, "—", "—", "—", "—", "—", "—", "—")

    if case_id not in cases:
        return (f"❌ Case '{case_id}' not found.", "—", "—", "—", "—", "—", "—", "—")

    case = cases[case_id]
    gt = _verdict_badge(case.ground_truth_is_fraud)
    trigger = case.trigger_label

    # --- Arrow 2: open a LangFuse trace for this UI investigation ---
    lf = _get_langfuse()
    trace = lf.trace(
        name="fraud-investigation-ui",
        input={
            "case_id": case.case_id,
            "seed_transaction_id": case.seed_transaction_id,
            "client_id": case.client_id,
            "card_id": case.card_id,
            "window_start": case.window_start,
            "window_end": case.window_end,
            "trigger": case.trigger_label,
            "ground_truth_is_fraud": case.ground_truth_is_fraud,
        },
        tags=[case.case_id, "gradio-ui"],
        metadata={"source": "gradio_app", "ground_truth": str(case.ground_truth_is_fraud)},
    )

    try:
        # Run the agent — returns RunResult (not just FraudAnalysisOutput)
        run_result: RunResult = asyncio.run(run_case(case))

        if run_result.output is None:
            trace.update(output={"error": "Agent produced no valid JSON output"})
            lf.flush()
            return (
                trigger, gt,
                "❌ Agent failed to produce a valid JSON verdict.",
                "—", "—", "—", "—", "—",
            )

        output = run_result.output
        match = output.is_fraud == case.ground_truth_is_fraud

        verdict = _verdict_badge(output.is_fraud)
        pattern = output.fraud_pattern.value.replace("_", " ").title()
        confidence = _confidence_bar(output.confidence_score)
        flagged = ", ".join(output.flagged_transaction_ids) if output.flagged_transaction_ids else "None"
        accuracy = _match_badge(match)

        # --- Arrow 2: update trace with agent output and log a score ---
        trace.update(
            output={
                "is_fraud": output.is_fraud,
                "fraud_pattern": output.fraud_pattern.value,
                "flagged_transaction_ids": output.flagged_transaction_ids,
                "confidence_score": output.confidence_score,
                "tools_called": run_result.tools_called,
                "check_accuracy_called": run_result.check_accuracy_called,
                "e2b_called": run_result.e2b_called,
                "sql_query_count": len(run_result.sql_queries),
            },
        )
        lf.score(
            trace_id=trace.id,
            name="ui_accuracy",
            value=1.0 if match else 0.0,
            comment=f"Predicted: {'FRAUD' if output.is_fraud else 'LEGIT'} | Ground truth: {'FRAUD' if case.ground_truth_is_fraud else 'LEGIT'}",
        )
        lf.score(
            trace_id=trace.id,
            name="confidence_score",
            value=output.confidence_score,
        )
        lf.flush()

        return (trigger, gt, verdict, pattern, confidence, flagged, output.explanation, accuracy)

    except Exception as e:
        trace.update(output={"error": str(e)})
        lf.flush()
        logger.exception("Agent run failed for case %s", case_id)
        return (trigger, gt, f"❌ Error: {e}", "—", "—", "—", "—", "—")


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def _get_case_choices() -> list[str]:
    cases = _load_cases()
    if not cases:
        return ["(no cases — run create-cases first)"]
    return sorted(cases.keys())


with gr.Blocks(title="Fraud Analytics Agent") as demo:
    gr.Markdown(
        """
# 🔍 Fraud Analytics Investigation Agent
Select a case from the dropdown and click **Investigate** to run the agent.
The agent uses SQL, Python analysis (E2B), and ground-truth label lookup to produce a verdict.
Every investigation is logged to LangFuse for observability.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            case_dropdown = gr.Dropdown(
                label="Case ID",
                choices=_get_case_choices(),
                interactive=True,
                info="Cases generated by `cli.py create-cases`",
            )
            refresh_btn = gr.Button("🔄 Refresh cases", size="sm", variant="secondary")
            run_btn = gr.Button("🔍 Investigate", variant="primary", size="lg")

        with gr.Column(scale=2):
            trigger_out = gr.Textbox(label="Trigger", interactive=False)
            gt_out = gr.Textbox(label="Ground Truth", interactive=False)

    gr.Markdown("---")
    gr.Markdown("### Agent Verdict")

    with gr.Row():
        verdict_out = gr.Textbox(label="Verdict", interactive=False, scale=1)
        pattern_out = gr.Textbox(label="Fraud Pattern", interactive=False, scale=1)
        confidence_out = gr.Textbox(label="Confidence", interactive=False, scale=1)
        accuracy_out = gr.Textbox(label="Accuracy vs Ground Truth", interactive=False, scale=1)

    flagged_out = gr.Textbox(label="Flagged Transaction IDs", interactive=False)
    explanation_out = gr.Textbox(
        label="Investigation Explanation",
        interactive=False,
        lines=12,
        max_lines=30,
    )

    run_btn.click(
        fn=run_investigation,
        inputs=[case_dropdown],
        outputs=[
            trigger_out,
            gt_out,
            verdict_out,
            pattern_out,
            confidence_out,
            flagged_out,
            explanation_out,
            accuracy_out,
        ],
    )

    refresh_btn.click(
        fn=lambda: gr.update(choices=_get_case_choices()),
        outputs=[case_dropdown],
    )

    gr.Markdown(
        """
---
**Tips:**
- Every investigation is logged to LangFuse automatically (Arrow 2 in the architecture).
- The agent calls `check_accuracy()` internally — accuracy is self-reported in the explanation.
- The **Accuracy vs Ground Truth** field shows the post-hoc comparison done by the UI.
        """
    )


if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
