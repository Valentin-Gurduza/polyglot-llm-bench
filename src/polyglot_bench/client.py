"""
Polyglot-LLM-Bench — Async OpenRouter API Client

Wraps the openai.AsyncOpenAI SDK pointed at OpenRouter's unified endpoint.
Includes exponential backoff retries, rate limiting, and structured error handling.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polyglot_bench.config import Settings
from polyglot_bench.models import ModelResponse, TokenUsage


class OpenRouterClient:
    """Async client for OpenRouter's unified LLM API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.execution.max_workers)

        self._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            default_headers={
                "HTTP-Referer": settings.app_referer,
                "X-Title": settings.app_title,
            },
            timeout=120.0,
        )

        # httpx client for raw API calls (model listing, etc.)
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "HTTP-Referer": settings.app_referer,
                "X-Title": settings.app_title,
            },
            timeout=30.0,
        )

    async def close(self) -> None:
        """Close underlying HTTP connections."""
        await self._client.close()
        await self._http.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Model Listing (raw)
    # ------------------------------------------------------------------
    async def list_models_raw(self) -> dict[str, Any]:
        """
        Fetch the raw model catalog from GET /api/v1/models.

        Returns the JSON response dict containing a 'data' array.
        Used by the discovery module for parsing and caching.
        """
        response = await self._http.get("/models")
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Chat Completion
    # ------------------------------------------------------------------
    async def complete(
        self,
        model: str,
        prompt_id: int,
        language: str,
        category: str,
        prompt_text: str,
    ) -> ModelResponse:
        """
        Send a chat completion request with rate limiting and retries.

        Returns a ModelResponse (always — errors are captured, not raised).
        """
        async with self._semaphore:
            return await self._complete_with_retry(
                model=model,
                prompt_id=prompt_id,
                language=language,
                category=category,
                prompt_text=prompt_text,
            )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    async def _complete_with_retry(
        self,
        model: str,
        prompt_id: int,
        language: str,
        category: str,
        prompt_text: str,
    ) -> ModelResponse:
        """Inner method with tenacity retry logic."""
        t0 = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self._settings.sampling.temperature,
                top_p=self._settings.sampling.top_p,
                max_tokens=self._settings.sampling.max_tokens,
            )

            latency_ms = (time.perf_counter() - t0) * 1000

            # Extract content
            content = ""
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content

            # Extract token usage
            usage = TokenUsage()
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0,
                )

            return ModelResponse(
                model=model,
                prompt_id=prompt_id,
                language=language,
                category=category,
                prompt_text=prompt_text,
                raw_response=content,
                latency_ms=latency_ms,
                token_usage=usage,
            )

        except (RateLimitError, APITimeoutError):
            # Let tenacity retry these
            raise

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return ModelResponse(
                model=model,
                prompt_id=prompt_id,
                language=language,
                category=category,
                prompt_text=prompt_text,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
