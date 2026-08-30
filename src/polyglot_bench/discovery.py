"""
Polyglot-LLM-Bench — Model Discovery & Search

Fetches, caches, and searches the OpenRouter model catalog.
Supports filtering by provider, pricing tier (free), modality, and context length.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from polyglot_bench.config import Settings


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------
class ModelInfo(BaseModel):
    """Parsed metadata for a single OpenRouter model."""

    id: str = Field(description="Model slug, e.g. 'openai/gpt-4o' or 'meta-llama/llama-4-maverick:free'")
    name: str = Field(default="", description="Human-readable name")
    description: str = Field(default="")
    context_length: int = Field(default=0, description="Maximum context window in tokens")
    pricing_prompt: str = Field(default="0", description="Cost per prompt token (string)")
    pricing_completion: str = Field(default="0", description="Cost per completion token (string)")
    modality: str = Field(default="text->text", description="e.g. 'text->text', 'text+image->text'")
    tokenizer: str = Field(default="")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_free(self) -> bool:
        """Whether this model is free (`:free` suffix or $0 pricing)."""
        if self.id.endswith(":free"):
            return True
        try:
            return float(self.pricing_prompt) == 0 and float(self.pricing_completion) == 0
        except (ValueError, TypeError):
            return False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def provider(self) -> str:
        """Extract provider prefix from model slug."""
        return self.id.split("/")[0] if "/" in self.id else self.id


# ---------------------------------------------------------------------------
# Catalog Cache Schema
# ---------------------------------------------------------------------------
class _CachedCatalog(BaseModel):
    """On-disk JSON cache format."""

    fetched_at: float = Field(description="Unix timestamp of fetch")
    models: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model Catalog
# ---------------------------------------------------------------------------
class ModelCatalog:
    """
    Fetch, cache, search, and filter the OpenRouter model catalog.

    Usage::

        catalog = ModelCatalog(settings)
        models = await catalog.get_catalog()
        free = catalog.filter_free(models)
        hits = catalog.search("llama", models)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache_path = Path(settings.discovery.cache_path)
        self._ttl = settings.discovery.cache_ttl_seconds
        self._models: list[ModelInfo] | None = None

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    async def fetch_models(self) -> list[ModelInfo]:
        """
        Fetch the full model catalog from OpenRouter's /api/v1/models endpoint.

        Uses the client's `list_models_raw()` (lazy-imported to avoid circular deps).
        """
        from polyglot_bench.client import OpenRouterClient

        async with OpenRouterClient(self._settings) as client:
            raw = await client.list_models_raw()

        return self._parse_raw_models(raw)

    @staticmethod
    def _parse_raw_models(raw: dict[str, Any]) -> list[ModelInfo]:
        """Parse the raw API JSON into ModelInfo objects."""
        models: list[ModelInfo] = []
        for item in raw.get("data", []):
            pricing = item.get("pricing", {})
            arch = item.get("architecture", {})
            models.append(
                ModelInfo(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    description=item.get("description", "")[:500],  # truncate long descriptions
                    context_length=item.get("context_length", 0),
                    pricing_prompt=str(pricing.get("prompt", "0")),
                    pricing_completion=str(pricing.get("completion", "0")),
                    modality=arch.get("modality", "text->text"),
                    tokenizer=arch.get("tokenizer", ""),
                )
            )
        return models

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def _read_cache(self) -> list[ModelInfo] | None:
        """Read cached catalog if it exists and is within TTL."""
        if not self._cache_path.exists():
            return None

        try:
            with open(self._cache_path) as f:
                cached = _CachedCatalog(**json.load(f))

            age = time.time() - cached.fetched_at
            if age > self._ttl:
                return None  # Expired

            return self._parse_raw_models({"data": cached.models})
        except Exception:
            return None

    def _write_cache(self, raw_data: list[dict[str, Any]]) -> None:
        """Write model data to the local cache file."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached = _CachedCatalog(fetched_at=time.time(), models=raw_data)
        with open(self._cache_path, "w") as f:
            json.dump(cached.model_dump(), f, indent=2)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def get_catalog(self, *, force_refresh: bool = False) -> list[ModelInfo]:
        """
        Get the model catalog, using cache when available.

        Args:
            force_refresh: If True, bypass cache and re-fetch from API.
        """
        if not force_refresh and self._models is not None:
            return self._models

        if not force_refresh:
            cached = self._read_cache()
            if cached is not None:
                self._models = cached
                return self._models

        # Fetch fresh
        from polyglot_bench.client import OpenRouterClient

        async with OpenRouterClient(self._settings) as client:
            raw = await client.list_models_raw()

        self._models = self._parse_raw_models(raw)

        # Write raw data to cache
        self._write_cache(raw.get("data", []))

        return self._models

    # ------------------------------------------------------------------
    # Search & Filters
    # ------------------------------------------------------------------
    @staticmethod
    def search(query: str, models: list[ModelInfo]) -> list[ModelInfo]:
        """
        Fuzzy substring search across model id, name, and description.

        Case-insensitive. Returns models matching ALL space-separated terms.
        """
        terms = query.lower().split()
        results: list[ModelInfo] = []
        for m in models:
            haystack = f"{m.id} {m.name} {m.description}".lower()
            if all(term in haystack for term in terms):
                results.append(m)
        return results

    @staticmethod
    def filter_free(models: list[ModelInfo]) -> list[ModelInfo]:
        """Return only free-tier models."""
        return [m for m in models if m.is_free]

    @staticmethod
    def filter_by_provider(provider: str, models: list[ModelInfo]) -> list[ModelInfo]:
        """Filter models by provider prefix (e.g. 'openai', 'anthropic')."""
        provider = provider.lower()
        return [m for m in models if m.provider.lower() == provider]

    @staticmethod
    def filter_by_modality(modality: str, models: list[ModelInfo]) -> list[ModelInfo]:
        """Filter by modality string (e.g. 'text->text')."""
        modality = modality.lower()
        return [m for m in models if modality in m.modality.lower()]

    @staticmethod
    def filter_by_min_context(min_ctx: int, models: list[ModelInfo]) -> list[ModelInfo]:
        """Filter models with at least `min_ctx` context window."""
        return [m for m in models if m.context_length >= min_ctx]

    @staticmethod
    def get_model(model_id: str, models: list[ModelInfo]) -> ModelInfo | None:
        """Look up a single model by exact ID."""
        for m in models:
            if m.id == model_id:
                return m
        return None


# ---------------------------------------------------------------------------
# Validation Helper
# ---------------------------------------------------------------------------
def validate_model_ids(
    ids: list[str], catalog: list[ModelInfo]
) -> tuple[list[str], list[str]]:
    """
    Validate a list of model IDs against the catalog.

    Returns:
        (valid_ids, invalid_ids)
    """
    catalog_ids = {m.id for m in catalog}
    valid: list[str] = []
    invalid: list[str] = []
    for mid in ids:
        if mid in catalog_ids:
            valid.append(mid)
        else:
            invalid.append(mid)
    return valid, invalid
