"""
Polyglot-LLM-Bench — Configuration Management

Loads settings from config.yaml, .env, and CLI overrides using Pydantic v2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Load .env as early as possible
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class SamplingConfig(BaseModel):
    """LLM sampling parameters for deterministic evaluation."""

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=128_000)


class ExecutionConfig(BaseModel):
    """Concurrency and retry settings."""

    max_workers: int = Field(default=5, ge=1, le=50)
    retry_attempts: int = Field(default=5, ge=0, le=20)
    retry_min_wait: int = Field(default=2, ge=1)
    retry_max_wait: int = Field(default=60, ge=1)
    rate_limit_rpm: int = Field(default=60, ge=1)


class DiscoveryConfig(BaseModel):
    """Model catalog discovery & caching settings."""

    cache_ttl_seconds: int = Field(default=3600, ge=0)
    cache_path: str = Field(default=".cache/models_catalog.json")
    include_free: bool = Field(default=True)


class OutputConfig(BaseModel):
    """Output file paths."""

    directory: str = Field(default="results")
    results_json: str = Field(default="benchmark_results.json")
    evaluation_csv: str = Field(default="evaluation_sheet.csv")
    leaderboard_csv: str = Field(default="leaderboard.csv")


class DatasetConfig(BaseModel):
    """Dataset file location."""

    path: str = Field(default="data/dataset.json")


# ---------------------------------------------------------------------------
# Main Settings
# ---------------------------------------------------------------------------
class Settings(BaseModel):
    """Root configuration for Polyglot-LLM-Bench."""

    languages: list[str] = Field(default=["ro", "en", "fr"])
    models: list[str] = Field(
        default=[
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-large",
            "deepseek/deepseek-chat",
        ]
    )
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # Loaded from environment — not in config.yaml
    api_key: str = Field(default="")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    app_referer: str = Field(default="https://github.com/polyglot-llm-bench")
    app_title: str = Field(default="Polyglot-LLM-Bench")

    @field_validator("languages", mode="before")
    @classmethod
    def _normalise_languages(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [lang.strip().lower() for lang in v.split(",") if lang.strip()]
        return [lang.strip().lower() for lang in v]

    @field_validator("models", mode="before")
    @classmethod
    def _normalise_models(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return [m.strip() for m in v]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_settings(
    config_path: str | Path = "config.yaml",
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> Settings:
    """
    Load settings by merging:
      1. Defaults (Pydantic)
      2. config.yaml
      3. Environment variables
      4. CLI overrides
    """
    data: dict[str, Any] = {}

    # --- Layer 1: config.yaml -------------------------------------------------
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                data = raw

    # --- Layer 2: Environment variables ---------------------------------------
    data["api_key"] = os.getenv("OPENROUTER_API_KEY", data.get("api_key", ""))
    data["base_url"] = os.getenv(
        "OPENROUTER_BASE_URL", data.get("base_url", "https://openrouter.ai/api/v1")
    )
    data["app_referer"] = os.getenv("APP_REFERER", data.get("app_referer", ""))
    data["app_title"] = os.getenv("APP_TITLE", data.get("app_title", "Polyglot-LLM-Bench"))

    # --- Layer 3: CLI overrides -----------------------------------------------
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                data[key] = value

    return Settings(**data)
