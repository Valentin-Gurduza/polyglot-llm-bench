"""
Polyglot-LLM-Bench — Benchmark Runner

Orchestrates the cross-product (models × prompts) evaluation pipeline.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from polyglot_bench.client import OpenRouterClient
from polyglot_bench.config import Settings
from polyglot_bench.discovery import ModelCatalog, validate_model_ids
from polyglot_bench.models import BenchmarkPrompt, BenchmarkRunMeta, ModelResponse

console = Console()


class BenchmarkRunner:
    """
    Orchestrates benchmark execution across models and prompts.

    1. Validates model IDs against the live OpenRouter catalog.
    2. Builds the (models × prompts) task matrix.
    3. Executes tasks concurrently with progress tracking.
    4. Returns collected ModelResponse objects.
    """

    def __init__(
        self,
        settings: Settings,
        prompts: list[BenchmarkPrompt],
        *,
        refresh_models: bool = False,
    ) -> None:
        self._settings = settings
        self._prompts = prompts
        self._refresh_models = refresh_models
        self._responses: list[ModelResponse] = []

    # ------------------------------------------------------------------
    # Pre-flight: Validate models
    # ------------------------------------------------------------------
    async def validate_models(self) -> list[str]:
        """
        Validate configured model IDs against the live OpenRouter catalog.

        Returns the list of valid model IDs. Prints warnings for invalid ones.
        """
        catalog = ModelCatalog(self._settings)

        try:
            models = await catalog.get_catalog(force_refresh=self._refresh_models)
        except Exception as exc:
            console.print(
                f"[yellow]⚠ Could not fetch model catalog: {exc}[/yellow]"
            )
            console.print("[dim]Proceeding with configured models without validation...[/dim]")
            return self._settings.models

        valid, invalid = validate_model_ids(self._settings.models, models)

        if invalid:
            console.print(f"\n[yellow]⚠ Unknown model IDs (not in OpenRouter catalog):[/yellow]")
            for mid in invalid:
                console.print(f"  [red]✗[/red] {mid}")
            console.print()

        if not valid:
            console.print("[red]✗ No valid models remaining. Aborting.[/red]")
            return []

        console.print(f"[green]✓ {len(valid)} model(s) validated against OpenRouter catalog[/green]")
        return valid

    # ------------------------------------------------------------------
    # Dry Run
    # ------------------------------------------------------------------
    def print_task_matrix(self, models: list[str]) -> None:
        """Print the task matrix as a Rich table."""
        table = Table(
            title="📋 Benchmark Task Matrix (Dry Run)",
            show_lines=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=5)
        table.add_column("Model", style="bold")
        table.add_column("Prompt ID", justify="center")
        table.add_column("Language", justify="center")
        table.add_column("Category", style="italic")
        table.add_column("Prompt (truncated)", max_width=50)

        idx = 0
        for model in models:
            for prompt in self._prompts:
                idx += 1
                table.add_row(
                    str(idx),
                    model,
                    str(prompt.id),
                    prompt.language.upper(),
                    prompt.category,
                    prompt.prompt[:80] + ("..." if len(prompt.prompt) > 80 else ""),
                )

        console.print(table)
        console.print(f"\n[bold]Total tasks:[/bold] {idx}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def run(self, *, dry_run: bool = False) -> tuple[list[ModelResponse], BenchmarkRunMeta]:
        """
        Execute the full benchmark pipeline.

        Args:
            dry_run: If True, print the task matrix and exit without API calls.

        Returns:
            Tuple of (responses list, run metadata).
        """
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc)

        # Validate models
        valid_models = await self.validate_models()
        if not valid_models:
            meta = BenchmarkRunMeta(
                run_id=run_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                languages=self._settings.languages,
                models=[],
            )
            return [], meta

        if dry_run:
            self.print_task_matrix(valid_models)
            meta = BenchmarkRunMeta(
                run_id=run_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                languages=self._settings.languages,
                models=valid_models,
                total_tasks=len(valid_models) * len(self._prompts),
            )
            return [], meta

        # Build task list
        tasks: list[dict[str, Any]] = []
        for model in valid_models:
            for prompt in self._prompts:
                tasks.append({"model": model, "prompt": prompt})

        console.print(
            f"\n[bold]🚀 Starting benchmark run [cyan]{run_id}[/cyan] "
            f"— {len(tasks)} tasks across {len(valid_models)} models[/bold]\n"
        )

        # Execute with progress tracking
        self._responses = []
        async with OpenRouterClient(self._settings) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("Benchmarking...", total=len(tasks))

                # Create coroutines
                async def _run_one(task_info: dict[str, Any]) -> ModelResponse:
                    prompt: BenchmarkPrompt = task_info["prompt"]
                    result = await client.complete(
                        model=task_info["model"],
                        prompt_id=prompt.id,
                        language=prompt.language,
                        category=prompt.category,
                        prompt_text=prompt.prompt,
                    )
                    progress.advance(task_id)
                    return result

                # Run concurrently (semaphore inside client handles rate limiting)
                results = await asyncio.gather(
                    *[_run_one(t) for t in tasks],
                    return_exceptions=True,
                )

        # Collect results
        for r in results:
            if isinstance(r, ModelResponse):
                self._responses.append(r)
            elif isinstance(r, Exception):
                console.print(f"[red]Unexpected error: {r}[/red]")

        # Summary
        successful = sum(1 for r in self._responses if r.success)
        failed = len(self._responses) - successful
        completed_at = datetime.now(timezone.utc)

        self._print_summary(valid_models, successful, failed)

        meta = BenchmarkRunMeta(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            languages=self._settings.languages,
            models=valid_models,
            total_tasks=len(tasks),
            successful_tasks=successful,
            failed_tasks=failed,
        )

        return self._responses, meta

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _print_summary(self, models: list[str], successful: int, failed: int) -> None:
        """Print a summary table after benchmark completion."""
        console.print("\n")
        table = Table(title="📊 Benchmark Summary", header_style="bold green")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Models evaluated", str(len(models)))
        table.add_row("Total prompts", str(len(self._prompts)))
        table.add_row("Total tasks", str(successful + failed))
        table.add_row("Successful", f"[green]{successful}[/green]")
        table.add_row("Failed", f"[red]{failed}[/red]" if failed else "[green]0[/green]")

        if self._responses:
            avg_latency = sum(r.latency_ms for r in self._responses) / len(self._responses)
            table.add_row("Avg latency", f"{avg_latency:.0f} ms")

        console.print(table)
