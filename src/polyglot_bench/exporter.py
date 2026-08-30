"""
Polyglot-LLM-Bench — Results Exporter

Generates evaluation CSVs, leaderboard tables, and raw JSON results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from polyglot_bench.models import (
    BenchmarkRunMeta,
    EvaluationRow,
    LeaderboardEntry,
    ModelResponse,
)


def _ensure_dir(path: Path) -> None:
    """Ensure the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# JSON Results
# ---------------------------------------------------------------------------
def export_results_json(
    responses: list[ModelResponse],
    meta: BenchmarkRunMeta,
    path: str | Path,
) -> Path:
    """
    Export the full benchmark results as JSON.

    Includes run metadata and all individual responses.
    """
    path = Path(path)
    _ensure_dir(path)

    output = {
        "meta": meta.model_dump(mode="json"),
        "responses": [r.model_dump(mode="json") for r in responses],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return path


# ---------------------------------------------------------------------------
# Evaluation Sheet CSV
# ---------------------------------------------------------------------------
def export_evaluation_sheet(
    responses: list[ModelResponse],
    path: str | Path,
) -> Path:
    """
    Generate an evaluation_sheet.csv pre-populated with model outputs
    and empty scoring columns for human annotators.
    """
    path = Path(path)
    _ensure_dir(path)

    rows: list[dict] = []
    for r in responses:
        row = EvaluationRow(
            prompt_id=r.prompt_id,
            language=r.language,
            category=r.category,
            model=r.model,
            prompt_text=r.prompt_text,
            model_response=r.raw_response if r.success else f"[ERROR] {r.error}",
            constraints="",
            constraint_adherence=None,
            linguistic_naturalness=None,
            factual_accuracy=None,
            tone_clarity=None,
            total_score=None,
            annotator_notes="",
        )
        rows.append(row.model_dump())

    df = pd.DataFrame(rows)

    # Reorder columns for annotator convenience
    column_order = [
        "prompt_id",
        "language",
        "category",
        "model",
        "prompt_text",
        "model_response",
        "constraints",
        "constraint_adherence",
        "linguistic_naturalness",
        "factual_accuracy",
        "tone_clarity",
        "total_score",
        "annotator_notes",
    ]
    df = df[[c for c in column_order if c in df.columns]]
    df.to_csv(path, index=False, encoding="utf-8")

    return path


# ---------------------------------------------------------------------------
# Leaderboard CSV
# ---------------------------------------------------------------------------
def export_leaderboard(
    responses: list[ModelResponse],
    path: str | Path,
) -> Path:
    """
    Generate a leaderboard.csv with aggregated per-model-per-language stats.
    """
    path = Path(path)
    _ensure_dir(path)

    # Group by (model, language)
    groups: dict[tuple[str, str], list[ModelResponse]] = {}
    for r in responses:
        key = (r.model, r.language)
        groups.setdefault(key, []).append(r)

    entries: list[dict] = []
    for (model, language), resps in sorted(groups.items()):
        successful = [r for r in resps if r.success]
        entry = LeaderboardEntry(
            model=model,
            language=language,
            total_prompts=len(resps),
            successful=len(successful),
            failed=len(resps) - len(successful),
            avg_latency_ms=(
                sum(r.latency_ms for r in successful) / len(successful)
                if successful
                else 0.0
            ),
            avg_prompt_tokens=(
                sum(r.token_usage.prompt_tokens for r in successful) / len(successful)
                if successful
                else 0.0
            ),
            avg_completion_tokens=(
                sum(r.token_usage.completion_tokens for r in successful) / len(successful)
                if successful
                else 0.0
            ),
        )
        entries.append(entry.model_dump())

    df = pd.DataFrame(entries)
    df.to_csv(path, index=False, encoding="utf-8")

    return path
