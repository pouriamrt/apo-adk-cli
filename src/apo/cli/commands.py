from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from apo.core.config import APOConfig

console = Console()


def _detect_default_model() -> str:
    """Pick the default model based on which API key is set."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-5.2"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini/gemini-2.5-flash"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-6"
    return "gemini/gemini-2.5-flash"


@click.group()
def cli():
    """APO — Automatic Prompt Optimizer"""
    load_dotenv()


@cli.command()
@click.option("--prompt", type=str, help="Prompt template string (must contain {input})")
@click.option("--prompt-file", type=click.Path(exists=True), help="Path to prompt template file")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to dataset (JSON/CSV)")
@click.option("--model", default=None, help="LiteLLM model string (auto-detected from API key if omitted)")
@click.option("--optimizer-model", default=None, help="Model for APO gradient/edit (defaults to --model)")
@click.option("--eval-mode", type=click.Choice(["auto", "reference", "llm-judge"]), default="auto")
@click.option("--beam-width", default=3, type=int, help="Beam search width")
@click.option("--beam-rounds", default=5, type=int, help="Number of optimization rounds")
@click.option("--n-runners", default=4, type=int, help="Parallel rollout runners")
@click.option("--output", "-o", type=click.Path(), help="Save optimized prompt to file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed rollout output")
def optimize(
    prompt: str | None,
    prompt_file: str | None,
    dataset: str,
    model: str | None,
    optimizer_model: str | None,
    eval_mode: str,
    beam_width: int,
    beam_rounds: int,
    n_runners: int,
    output: str | None,
    verbose: bool,
):
    """Optimize a prompt using APO algorithm."""
    prompt_text = _load_prompt(prompt, prompt_file)

    if not model:
        model = _detect_default_model()

    # Derive optimizer model from rollout model when not specified
    # e.g. "gemini/gemini-2.5-flash" -> "gemini-2.5-flash"
    # e.g. "openai/gpt-4o" -> "gpt-4o"
    if not optimizer_model:
        optimizer_model = model.split("/", 1)[-1] if "/" in model else model

    config = APOConfig(
        model=model,
        optimizer_model=optimizer_model,
        beam_width=beam_width,
        beam_rounds=beam_rounds,
        n_runners=n_runners,
        eval_mode=eval_mode,
        verbose=verbose,
    )

    from apo.core.optimizer import run_optimization

    try:
        best_prompt, initial_score, best_score = run_optimization(
            prompt_text, Path(dataset), config
        )
    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {e}")
        raise SystemExit(1)

    console.print("\n[bold green]Optimization complete![/bold green]\n")
    console.print("[bold]Best prompt:[/bold]")
    console.print(best_prompt)

    if output:
        Path(output).write_text(best_prompt)
        console.print(f"\nSaved to: {output}")


@cli.command()
@click.option("--prompt", type=str, help="Prompt template string")
@click.option("--prompt-file", type=click.Path(exists=True), help="Path to prompt template file")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to dataset (JSON/CSV)")
@click.option("--model", default=None, help="LiteLLM model string (auto-detected from API key if omitted)")
@click.option("--eval-mode", type=click.Choice(["auto", "reference", "llm-judge"]), default="auto")
@click.option("--verbose", "-v", is_flag=True)
def evaluate(
    prompt: str | None,
    prompt_file: str | None,
    dataset: str,
    model: str | None,
    eval_mode: str,
    verbose: bool,
):
    """Evaluate a prompt on a dataset (no optimization)."""
    prompt_text = _load_prompt(prompt, prompt_file)

    if not model:
        model = _detect_default_model()

    config = APOConfig(model=model, eval_mode=eval_mode, verbose=verbose)

    from apo.core.optimizer import run_evaluation

    try:
        avg_score = run_evaluation(prompt_text, Path(dataset), config)
    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {e}")
        raise SystemExit(1)

    console.print(f"\n[bold]Average score:[/bold] {avg_score:.3f}")


def _load_prompt(prompt: str | None, prompt_file: str | None) -> str:
    """Load prompt from string or file."""
    if prompt and prompt_file:
        raise click.UsageError("Provide either --prompt or --prompt-file, not both")
    if not prompt and not prompt_file:
        raise click.UsageError("Provide either --prompt or --prompt-file")

    if prompt_file:
        text = Path(prompt_file).read_text().strip()
    else:
        text = prompt  # type: ignore

    if "{input}" not in text:
        raise click.UsageError("Prompt template must contain {input} placeholder")

    return text


if __name__ == "__main__":
    cli()
