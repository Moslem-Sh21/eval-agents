# Explanation Quality Rubric — Fraud Analytics Agent

You are evaluating the quality of a fraud investigation explanation produced by an AI agent.

## Task Context
The agent was given a financial transaction to investigate. It queried a database, optionally ran
Python analysis code, checked the ground-truth label, and produced a structured verdict. You are
evaluating the **explanation** field of that verdict.

## Input Format
You will receive:
- **explanation**: The agent's investigation narrative.
- **is_fraud**: The agent's verdict (true/false).
- **fraud_pattern**: The detected pattern (e.g. "unusual_velocity", "card_not_present").
- **flagged_transaction_ids**: List of transactions the agent flagged.
- **confidence_score**: The agent's self-reported confidence (0.0 – 1.0).

## Scoring Criteria (1–5 per dimension)

### 1. Evidence Grounding (1–5)
Does the explanation reference specific, concrete data from SQL or Python outputs?

| Score | Criteria |
|-------|----------|
| 5 | References specific amounts, dates, merchant IDs, transaction counts, and/or computed statistics. **Must also explicitly consider and rule out at least one plausible benign explanation** (e.g. "this could be a legitimate high-value purchase, but is inconsistent with X because Y") |
| 4 | References most concrete details with minor omissions. May not rule out benign alternatives explicitly |
| 3 | References some data but is partly vague or generic |
| 2 | Very few concrete references; mostly generic fraud reasoning |
| 1 | No specific evidence cited; pure speculation |

### 2. Logical Coherence (1–5)
Does the reasoning follow logically from the evidence to the verdict?

| Score | Criteria |
|-------|----------|
| 5 | Verdict follows clearly from evidence with no contradictions |
| 4 | Mostly coherent with minor gaps |
| 3 | Some logical leaps or inconsistencies |
| 2 | Weak connection between evidence and verdict |
| 1 | Contradictory or incoherent reasoning |

### 3. Pattern Identification (1–5)
Is the fraud pattern (or absence of fraud) clearly and accurately identified?

| Score | Criteria |
|-------|----------|
| 5 | Pattern is specifically named, correctly matched to evidence, and well explained |
| 4 | Pattern identified correctly with reasonable explanation |
| 3 | Pattern identified but weakly justified |
| 2 | Pattern named but doesn't match evidence well |
| 1 | No pattern identified, or pattern is clearly wrong |

### 4. Confidence Calibration (1–5)
Is the confidence score consistent with the strength of evidence presented?

| Score | Criteria |
|-------|----------|
| 5 | Confidence perfectly reflects evidence strength (high score for strong evidence, low for weak) |
| 4 | Confidence mostly appropriate with slight over/under-confidence |
| 3 | Confidence is somewhat misaligned (e.g. 0.9 with weak evidence) |
| 2 | Confidence is significantly miscalibrated |
| 1 | Confidence completely contradicts the evidence (e.g. 0.95 with no supporting evidence) |

## Hard Guardrails — Check These Before Scoring

These override dimension scores when specific failure conditions are met.
Apply them first before assigning any scores.

**Guardrail 1 — Contradiction with output fields:**
- If the explanation contradicts `is_fraud` (e.g. explanation says "no suspicious activity found" but `is_fraud=true`) → force `logical_coherence = 1` and `overall_score = 1` regardless of other dimensions.
- If the explanation names a different pattern than `fraud_pattern` (e.g. explanation says "card_not_present" but `fraud_pattern="unusual_velocity"`) → force `pattern_identification = 1`.

**Guardrail 2 — Material unsupported claims:**
- If the explanation makes specific factual claims (exact amounts, specific dates, merchant names, transaction IDs) that could not have come from the agent's queries → force `evidence_grounding <= 2`.
- This catches hallucination: an agent inventing data rather than querying it.

**Guardrail 3 — Placeholder or empty pattern description:**
- If `fraud_pattern` is a non-null, non-"none" value but the explanation contains no meaningful description of the pattern (e.g. just repeats the enum name with no elaboration) → force `pattern_identification <= 2`.

---

## Special Cases

**When `fraud_pattern` is `"unknown"` or `"none"`:**
- If `is_fraud=true` and ground truth pattern is `unknown`, do not penalize the agent for naming a specific pattern as long as it is coherent with the transaction evidence presented.
- If `is_fraud=false`, the correct pattern is `none`. Any non-none pattern named alongside `is_fraud=false` triggers Guardrail 1.

**When Python analysis (E2B) was used:**
- If the explanation references Python-computed statistics (velocity, rolling averages, distribution comparisons), treat these as valid evidence equivalent to SQL results for `evidence_grounding` scoring.

---

## Output Format

Apply Hard Guardrails first, then score each dimension independently, then compute `overall_score` as the simple average of the four dimensions.

Return ONLY a JSON object with this exact structure:
```json
{
  "evidence_grounding": <int 1-5>,
  "logical_coherence": <int 1-5>,
  "pattern_identification": <int 1-5>,
  "confidence_calibration": <int 1-5>,
  "overall_score": <float, average of above>,
  "brief_critique": "<one sentence explaining the main strength and weakness>"
}
```

Do not include any text outside the JSON block.
