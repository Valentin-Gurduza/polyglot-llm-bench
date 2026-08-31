#!/usr/bin/env python3
"""
Polyglot-LLM-Bench — CLI Entry Point

Subcommands:
    run             Execute the benchmark pipeline
    list-models     Browse the OpenRouter model catalog
    search-models   Search models by keyword
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Ensure src/ is importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from polyglot_bench import __version__
from polyglot_bench.config import load_settings
from polyglot_bench.dataset import filter_by_language, load_dataset
from polyglot_bench.discovery import ModelCatalog, ModelInfo
from polyglot_bench.exporter import (
    export_evaluation_sheet,
    export_leaderboard,
    export_per_model_evaluation_sheets,
    export_per_model_html_reports,
    export_per_model_results_json,
    export_results_json,
)
from polyglot_bench.runner import BenchmarkRunner

console = Console()

BANNER = r"""
[bold cyan]
  ____       _             _       _       _     _     __  __       ____                  _
 |  _ \ ___ | |_   _  __ _| | ___ | |_    | |   | |   |  \/  |     | __ )  ___ _ __   ___| |__
 | |_) / _ \| | | | |/ _` | |/ _ \| __|__ | |   | |   | |\/| |_____|  _ \ / _ \ '_ \ / __| '_ \
 |  __/ (_) | | |_| | (_| | | (_) | |_|__|| |___| |___| |  | |_____| |_) |  __/ | | | (__| | | |
 |_|   \___/|_|\__, |\__, |_|\___/ \__|   |_____|_____|_|  |_|     |____/ \___|_| |_|\___|_| |_|
               |___/ |___/
[/bold cyan]
[dim]Multi-Language Linguistic & Instruction-Following Benchmark via OpenRouter API[/dim]
[dim]Version {version}[/dim]
"""


# ═══════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyglot-bench",
        description="Polyglot-LLM-Bench: Multi-Language LLM Benchmark",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ─────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Execute the benchmark pipeline")
    run_parser.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated ISO 639-1 language codes (default: from config)",
    )
    run_parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated OpenRouter model slugs (default: from config)",
    )
    run_parser.add_argument(
        "--free-only", action="store_true",
        help="Auto-select all available :free models instead of --models",
    )
    run_parser.add_argument(
        "--workers", type=int, default=None,
        help="Max concurrent API requests (default: 5)",
    )
    run_parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    run_parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/)",
    )
    run_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show task matrix without executing API calls",
    )
    run_parser.add_argument(
        "--refresh-models", action="store_true",
        help="Force-refresh the model catalog cache before validation",
    )

    # ── list-models ────────────────────────────────────────────────────
    list_parser = subparsers.add_parser("list-models", help="Browse the OpenRouter model catalog")
    list_parser.add_argument(
        "--free-only", action="store_true",
        help="Show only free-tier models",
    )
    list_parser.add_argument(
        "--provider", type=str, default=None,
        help="Filter by provider (e.g. openai, anthropic, meta-llama)",
    )
    list_parser.add_argument(
        "--min-context", type=int, default=None,
        help="Minimum context window size",
    )
    list_parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output raw JSON instead of Rich table",
    )
    list_parser.add_argument(
        "--refresh", action="store_true",
        help="Bypass cache and fetch fresh catalog",
    )
    list_parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file",
    )

    # ── search-models ──────────────────────────────────────────────────
    search_parser = subparsers.add_parser("search-models", help="Search models by keyword")
    search_parser.add_argument(
        "query", type=str,
        help="Search query (fuzzy match on id, name, description)",
    )
    search_parser.add_argument(
        "--free-only", action="store_true",
        help="Show only free-tier models",
    )
    search_parser.add_argument(
        "--provider", type=str, default=None,
        help="Filter by provider",
    )
    search_parser.add_argument(
        "--min-context", type=int, default=None,
        help="Minimum context window size",
    )
    search_parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output raw JSON instead of Rich table",
    )
    search_parser.add_argument(
        "--refresh", action="store_true",
        help="Bypass cache and fetch fresh catalog",
    )
    search_parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file",
    )

    return parser


# ═══════════════════════════════════════════════════════════════════════════
# Display Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _format_price(price_str: str) -> str:
    """Format per-token price to $/M (per million tokens)."""
    try:
        per_token = float(price_str)
        if per_token == 0:
            return "[green]Free[/green]"
        per_million = per_token * 1_000_000
        if per_million < 0.01:
            return f"${per_million:.4f}"
        return f"${per_million:.2f}"
    except (ValueError, TypeError):
        return price_str


def display_models_table(models: list[ModelInfo], title: str = "OpenRouter Models") -> None:
    """Render a Rich table of models."""
    table = Table(title=title, show_lines=False, header_style="bold cyan", row_styles=["", "dim"])
    table.add_column("Model ID", style="bold", max_width=50)
    table.add_column("Name", max_width=35)
    table.add_column("Context", justify="right")
    table.add_column("Prompt $/M", justify="right")
    table.add_column("Completion $/M", justify="right")
    table.add_column("Free?", justify="center")
    table.add_column("Modality", max_width=20)

    for m in models:
        table.add_row(
            m.id,
            m.name[:35],
            f"{m.context_length:,}",
            _format_price(m.pricing_prompt),
            _format_price(m.pricing_completion),
            "[green]✓[/green]" if m.is_free else "[dim]–[/dim]",
            m.modality,
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(models)} model(s)[/dim]")


def display_models_json(models: list[ModelInfo]) -> None:
    """Output models as JSON."""
    data = [m.model_dump() for m in models]
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand Handlers
# ═══════════════════════════════════════════════════════════════════════════
async def handle_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""
    # Build CLI overrides
    overrides: dict = {}
    if args.languages:
        overrides["languages"] = args.languages
    if args.models:
        overrides["models"] = args.models
    if args.workers:
        overrides["execution"] = {"max_workers": args.workers}
    if args.output:
        overrides["output"] = {"directory": args.output}

    settings = load_settings(args.config, cli_overrides=overrides)

    # Handle --free-only: override models with free catalog
    if args.free_only:
        console.print("[cyan]🔍 Fetching free models from OpenRouter catalog...[/cyan]")
        catalog = ModelCatalog(settings)
        all_models = await catalog.get_catalog(force_refresh=args.refresh_models)
        free_models = ModelCatalog.filter_free(all_models)
        if not free_models:
            console.print("[red]✗ No free models found in the catalog.[/red]")
            return
        settings.models = [m.id for m in free_models]
        console.print(f"[green]✓ Found {len(free_models)} free model(s)[/green]")

    # Validate API key
    if not settings.api_key or settings.api_key.startswith("sk-or-v1-REPLACE"):
        console.print(
            "[red]✗ OPENROUTER_API_KEY not set or still placeholder.[/red]\n"
            "[dim]  Copy .env.example to .env and add your key from https://openrouter.ai/keys[/dim]"
        )
        return

    # Load dataset
    try:
        prompts = load_dataset(settings.dataset.path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]✗ Dataset error: {exc}[/red]")
        return

    # Filter by language
    prompts = filter_by_language(prompts, settings.languages)
    if not prompts:
        console.print(
            f"[red]✗ No prompts found for languages: {settings.languages}[/red]"
        )
        return

    console.print(
        f"[green]✓ Loaded {len(prompts)} prompt(s) for languages: "
        f"{', '.join(settings.languages)}[/green]"
    )

    # Run benchmark
    runner = BenchmarkRunner(
        settings, prompts, refresh_models=args.refresh_models
    )
    responses, meta = await runner.run(dry_run=args.dry_run)

    if args.dry_run or not responses:
        return

    # Export results directory
    out_dir = Path(settings.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export consolidated results
    json_path = export_results_json(
        responses, meta, out_dir / settings.output.results_json
    )
    csv_path = export_evaluation_sheet(
        responses, out_dir / settings.output.evaluation_csv
    )
    lb_path = export_leaderboard(
        responses, out_dir / settings.output.leaderboard_csv
    )

    # Export separate per-model results, CSV evaluation sheets, and HTML visual reports
    per_model_csvs = export_per_model_evaluation_sheets(responses, out_dir)
    per_model_jsons = export_per_model_results_json(responses, meta, out_dir)
    per_model_htmls = export_per_model_html_reports(responses, out_dir)

    console.print(f"\n[bold green]📁 Results exported to {out_dir}/ :[/bold green]")
    console.print(f"  [cyan]📊 Summary Leaderboard:[/cyan]  {lb_path}")
    console.print(f"  [cyan]📑 Master Eval Sheet:[/cyan]    {csv_path}")
    console.print(f"  [cyan]📦 Master Raw JSON:[/cyan]      {json_path}")

    if per_model_htmls:
        console.print(f"\n[bold green]🌐 Visual HTML Reports (Open in Browser):[/bold green]")
        for p in per_model_htmls:
            console.print(f"  • {p}")

    if per_model_csvs:
        console.print(f"\n[bold cyan]📊 Per-Model CSV Evaluation Sheets (for human annotators):[/bold cyan]")
        for p in per_model_csvs:
            console.print(f"  • {p}")

    if per_model_jsons:
        console.print(f"\n[bold cyan]📦 Per-Model Raw JSON Results:[/bold cyan]")
        for p in per_model_jsons:
            console.print(f"  • {p}")


async def handle_list_models(args: argparse.Namespace) -> None:
    """Handle the 'list-models' subcommand."""
    settings = load_settings(args.config)

    if not settings.api_key or settings.api_key.startswith("sk-or-v1-REPLACE"):
        console.print(
            "[red]✗ OPENROUTER_API_KEY not set.[/red]\n"
            "[dim]  Set it in .env to query the OpenRouter model catalog.[/dim]"
        )
        return

    catalog = ModelCatalog(settings)
    models = await catalog.get_catalog(force_refresh=args.refresh)

    # Apply filters
    if args.free_only:
        models = ModelCatalog.filter_free(models)
    if args.provider:
        models = ModelCatalog.filter_by_provider(args.provider, models)
    if args.min_context:
        models = ModelCatalog.filter_by_min_context(args.min_context, models)

    if not models:
        console.print("[yellow]No models match the specified filters.[/yellow]")
        return

    if args.output_json:
        display_models_json(models)
    else:
        title = "OpenRouter Models"
        filters = []
        if args.free_only:
            filters.append("free only")
        if args.provider:
            filters.append(f"provider={args.provider}")
        if args.min_context:
            filters.append(f"min_context={args.min_context:,}")
        if filters:
            title += f" ({', '.join(filters)})"
        display_models_table(models, title=title)


async def handle_search_models(args: argparse.Namespace) -> None:
    """Handle the 'search-models' subcommand."""
    settings = load_settings(args.config)

    if not settings.api_key or settings.api_key.startswith("sk-or-v1-REPLACE"):
        console.print(
            "[red]✗ OPENROUTER_API_KEY not set.[/red]\n"
            "[dim]  Set it in .env to query the OpenRouter model catalog.[/dim]"
        )
        return

    catalog = ModelCatalog(settings)
    all_models = await catalog.get_catalog(force_refresh=args.refresh)

    # Search
    models = ModelCatalog.search(args.query, all_models)

    # Apply additional filters
    if args.free_only:
        models = ModelCatalog.filter_free(models)
    if args.provider:
        models = ModelCatalog.filter_by_provider(args.provider, models)
    if args.min_context:
        models = ModelCatalog.filter_by_min_context(args.min_context, models)

    if not models:
        console.print(f'[yellow]No models match query "{args.query}" with the specified filters.[/yellow]')
        return

    if args.output_json:
        display_models_json(models)
    else:
        display_models_table(models, title=f'Search: "{args.query}"')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    console.print(BANNER.format(version=__version__))

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    handler = {
        "run": handle_run,
        "list-models": handle_list_models,
        "search-models": handle_search_models,
    }.get(args.command)

    if handler is None:
        parser.print_help()
        return

    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
