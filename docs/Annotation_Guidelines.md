# 📋 Polyglot-LLM-Bench — Human Annotation Guidelines

## Overview

This document provides standardized scoring criteria for human annotators evaluating LLM outputs in the **Polyglot-LLM-Bench** framework. All responses are scored across **4 dimensions** using a **1–5 scale**.

The evaluation sheet (`evaluation_sheet.csv`) is pre-populated with model outputs. Your task is to fill in the scoring columns and provide brief annotator notes.

---

## Scoring Dimensions

### 1. Constraint Adherence (1–5)

> Does the model's response satisfy all explicit constraints stated in the prompt?

| Score | Label | Criteria |
|:-----:|:------|:---------|
| **5** | Perfect Compliance | All constraints fully satisfied. No violations whatsoever. |
| **4** | Minor Deviation | All major constraints met; one minor, borderline violation (e.g., 101 words vs. 100-word limit). |
| **3** | Partial Compliance | Most constraints met (≥50%), but one or more clearly violated. |
| **2** | Significant Failure | Fewer than half of the constraints satisfied. Major violations present. |
| **1** | Complete Disregard | Response ignores most or all stated constraints. |

**Examples:**
- **Score 5:** Prompt asks for exactly 3 sentences without "good" → Response has exactly 3 sentences and never uses "good".
- **Score 3:** Same prompt → Response has 4 sentences but avoids "good".
- **Score 1:** Same prompt → Response is a paragraph with "good" used twice.

---

### 2. Linguistic Naturalness & Grammar/Orthography (1–5)

> Is the response fluent, grammatically correct, and natural-sounding in the target language?

| Score | Label | Criteria |
|:-----:|:------|:---------|
| **5** | Native-Speaker Fluency | Reads as if written by an educated native speaker. Impeccable grammar, orthography, and idiomatic expression. |
| **4** | Near-Native | Very fluent with minor stylistic imperfections. No grammar errors. Reads naturally. |
| **3** | Acceptable | Understandable but contains noticeable grammar errors, awkward phrasing, or non-idiomatic constructions. |
| **2** | Below Standard | Frequent grammatical errors, unnatural syntax, or misused idioms that hinder comprehension. |
| **1** | Incomprehensible | Severely broken language, machine-translation artifacts, or incoherent text. |

**Language-Specific Notes:**
- **Romanian (ro):** Check for correct diacritics (ă, â, î, ș, ț), proper case endings, and Romanian-specific idioms.
- **French (fr):** Check for accent marks (é, è, ê, ë, à, ç), gender agreement, and subjunctive usage where required.
- **English (en):** Check for consistent dialect (US/UK), proper article usage, and idiomatic phrasing.

---

### 3. Factual Accuracy & Hallucination Avoidance (1–5)

> Are all stated facts, dates, names, and claims verifiably correct?

| Score | Label | Criteria |
|:-----:|:------|:---------|
| **5** | Perfectly Accurate | All facts verifiable and precise. No hallucinations. Dates, names, and figures are correct. |
| **4** | Mostly Accurate | All key facts correct; one minor inaccuracy that doesn't change the overall message (e.g., approximate date off by 1 year). |
| **3** | Mixed Accuracy | Contains both correct and incorrect facts. No major fabrications, but some claims are unverifiable or misleading. |
| **2** | Significant Errors | Multiple factual errors or one major hallucination (e.g., invented historical event, wrong country). |
| **1** | Hallucinated Content | Predominantly fabricated or fictitious claims presented as fact. Highly unreliable. |

**Evaluation Protocol:**
1. Cross-reference specific dates, names, and figures against the `reference_notes` in the dataset.
2. Use established sources (Wikipedia, official government sites) for verification when reference_notes are insufficient.
3. Mark any claim you cannot verify as "unverifiable" in your annotator notes.

---

### 4. Tone, Clarity & User Helpfulness (1–5)

> Is the response appropriately toned, clearly structured, and genuinely helpful to the end user?

| Score | Label | Criteria |
|:-----:|:------|:---------|
| **5** | Exemplary | Perfectly matched tone (formal/informal as requested). Crystal-clear structure. Directly and completely addresses the user's need. |
| **4** | Very Good | Appropriate tone and mostly clear. Addresses the core need with minor room for improvement in structure or completeness. |
| **3** | Adequate | Acceptable tone but could be clearer. Partially addresses the user's need. Some structural issues. |
| **2** | Mismatched | Tone is inappropriate for the context (e.g., overly casual when formal was requested). Unclear organization. Missing key information. |
| **1** | Unhelpful | Completely wrong tone. Confusing, rambling, or off-topic. Does not address the user's actual need. |

---

## Total Score Calculation

$$\text{Total Score} = \text{Constraint Adherence} + \text{Linguistic Naturalness} + \text{Factual Accuracy} + \text{Tone \& Clarity}$$

| Total Score Range | Overall Quality |
|:-----------------:|:----------------|
| **18–20** | 🟢 Excellent |
| **14–17** | 🟡 Good |
| **10–13** | 🟠 Acceptable |
| **6–9** | 🔴 Below Standard |
| **4–5** | ⛔ Unacceptable |

---

## Annotator Notes Guidelines

For each evaluation, provide a brief note (1–3 sentences) covering:

1. **What was done well** — Strengths of the response.
2. **What was done poorly** — Specific failures (cite the exact constraint or fact).
3. **Borderline decisions** — If any score was a close call, explain your reasoning.

**Example annotator note:**
> "Constraint adherence: Response used 4 sentences instead of 3 (scored 3). Language quality is native-level Romanian with correct diacritics. Factual content about 1989 revolution is accurate — all dates match reference. Tone is appropriate for an educational explanation."

---

## Inter-Annotator Agreement (IAA) Protocol

When multiple annotators evaluate the same response:

1. **Independent scoring** — Each annotator scores independently before discussion.
2. **Disagreement threshold** — If any dimension differs by ≥2 points between annotators, a reconciliation discussion is required.
3. **Reconciliation** — Annotators discuss the specific disagreement and agree on a final score. Document the rationale.
4. **IAA metric** — Report Cohen's κ (kappa) or Krippendorff's α (alpha) for each dimension in the final report.

---

## Category-Specific Evaluation Guidance

### Constraint Adherence Prompts
- Count words, sentences, and characters precisely (use tools if needed).
- Forbidden-word checks must be case-insensitive.
- Schema/format constraints (numbered lists, bullet points) must match exactly.

### Cultural Localization Prompts
- Evaluate cultural appropriateness for the target audience.
- Translations of cultural references should feel natural, not literal.
- Penalize responses that simply translate words without adapting cultural context.

### Fact-Checking Prompts
- Cross-reference ALL dates, names, and figures against the provided `reference_notes`.
- Any invented/hallucinated fact = automatic ≤2 on Factual Accuracy.
- "Approximately correct" is acceptable only when labeled as approximate.

### Technical Localization Prompts
- Verify that standardized terminology is used (not anglicisms when native terms exist).
- Technical accuracy must be preserved even in simplified explanations.
- Evaluate whether the target audience would understand the response.

### Logical Reasoning Prompts
- Verify the final answer is correct.
- Check that intermediate reasoning steps are valid.
- A correct answer with flawed reasoning should score ≤3 on Constraint Adherence.

---

## Workflow Checklist

- [ ] Read the original prompt and constraints carefully.
- [ ] Read the model's response completely before scoring.
- [ ] Score each dimension independently (don't let one dimension influence another).
- [ ] Write annotator notes for every response.
- [ ] Flag any response you're unsure about for secondary review.
- [ ] Save your work frequently.

---

*Document version: 1.0 — Polyglot-LLM-Bench*
