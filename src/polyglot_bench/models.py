"""
Polyglot-LLM-Bench — Pydantic Data Models

Defines the core data structures used across the benchmark pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Benchmark Prompt (dataset schema)
# ---------------------------------------------------------------------------
class BenchmarkPrompt(BaseModel):
    """A single benchmark prompt from dataset.json."""

    id: int
    language: str = Field(description="ISO 639-1 language code")
    category: str = Field(description="Evaluation category")
    prompt: str = Field(description="The instruction/prompt text")
    constraints: list[str] = Field(default_factory=list, description="Verifiable constraints")
    reference_notes: str = Field(default="", description="Notes for annotators")


# ---------------------------------------------------------------------------
# Token Usage
# ---------------------------------------------------------------------------
class TokenUsage(BaseModel):
    """Token counts from a completion response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Model Response
# ---------------------------------------------------------------------------
class ModelResponse(BaseModel):
    """The result of a single (model × prompt) evaluation."""

    model: str = Field(description="OpenRouter model slug")
    prompt_id: int
    language: str
    category: str
    prompt_text: str
    raw_response: str = Field(default="", description="Model's raw text output")
    latency_ms: float = Field(default=0.0, description="Request latency in milliseconds")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = Field(default=None, description="Error message if request failed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success(self) -> bool:
        """Whether the request completed without error."""
        return self.error is None and len(self.raw_response) > 0


# ---------------------------------------------------------------------------
# Evaluation Row (for annotation CSV)
# ---------------------------------------------------------------------------
class EvaluationRow(BaseModel):
    """A row in the evaluation_sheet.csv — pre-populated + empty scoring columns."""

    prompt_id: int
    language: str
    category: str
    model: str
    prompt_text: str
    model_response: str
    constraints: str = Field(default="", description="JSON-encoded constraints list")

    # Scoring columns (to be filled by human annotators)
    constraint_adherence: int | None = Field(default=None, ge=1, le=5)
    linguistic_naturalness: int | None = Field(default=None, ge=1, le=5)
    factual_accuracy: int | None = Field(default=None, ge=1, le=5)
    tone_clarity: int | None = Field(default=None, ge=1, le=5)
    total_score: int | None = Field(default=None, ge=4, le=20)
    annotator_notes: str = Field(default="")


# ---------------------------------------------------------------------------
# Leaderboard Entry
# ---------------------------------------------------------------------------
class LeaderboardEntry(BaseModel):
    """Aggregated stats for a single model on the leaderboard."""

    model: str
    language: str
    total_prompts: int = 0
    successful: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success_rate(self) -> float:
        """Fraction of successful completions."""
        if self.total_prompts == 0:
            return 0.0
        return self.successful / self.total_prompts


# ---------------------------------------------------------------------------
# Benchmark Run Metadata
# ---------------------------------------------------------------------------
class BenchmarkRunMeta(BaseModel):
    """Metadata for a complete benchmark run."""

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    languages: list[str]
    models: list[str]
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
