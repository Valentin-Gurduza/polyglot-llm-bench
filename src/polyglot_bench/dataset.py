"""
Polyglot-LLM-Bench — Dataset Loader

Loads, validates, and filters the benchmark dataset from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from polyglot_bench.models import BenchmarkPrompt


def load_dataset(path: str | Path) -> list[BenchmarkPrompt]:
    """
    Load and validate the benchmark dataset from a JSON file.

    Args:
        path: Path to dataset.json.

    Returns:
        List of validated BenchmarkPrompt objects.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If any prompt fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(raw).__name__}")

    prompts: list[BenchmarkPrompt] = []
    for i, item in enumerate(raw):
        try:
            prompts.append(BenchmarkPrompt(**item))
        except Exception as exc:
            raise ValueError(f"Invalid prompt at index {i}: {exc}") from exc

    return prompts


def filter_by_language(
    prompts: list[BenchmarkPrompt], languages: list[str]
) -> list[BenchmarkPrompt]:
    """Filter prompts to only include the specified languages."""
    lang_set = {lang.lower() for lang in languages}
    return [p for p in prompts if p.language.lower() in lang_set]


def filter_by_category(
    prompts: list[BenchmarkPrompt], categories: list[str]
) -> list[BenchmarkPrompt]:
    """Filter prompts to only include the specified categories."""
    cat_set = {cat.lower() for cat in categories}
    return [p for p in prompts if p.category.lower() in cat_set]
