from __future__ import annotations

import os
from pathlib import Path

from openai import AsyncOpenAI
from rich.console import Console

from agentlightning import Trainer, PromptTemplate
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO

from apo.core.config import APOConfig
from apo.core.rollout import PromptTask, prompt_rollout, configure_rollout
from apo.data.loader import load_dataset, split_dataset, Task
from apo.evaluation.scorer import get_scorer

console = Console()


def _build_openai_client(config: APOConfig) -> AsyncOpenAI:
    """Build an AsyncOpenAI client for APO gradient/edit models.

    Uses Gemini's OpenAI-compatible API if a Google API key is available,
    otherwise falls back to OpenAI.
    """
    google_key = os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if google_key and config.optimizer_model.startswith("gemini"):
        return AsyncOpenAI(
            api_key=google_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    if openai_key:
        return AsyncOpenAI(api_key=openai_key)

    if google_key:
        return AsyncOpenAI(
            api_key=google_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    raise RuntimeError(
        "No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in your environment.\n"
        "See .env.example for details."
    )


def _tasks_to_prompt_tasks(tasks: list[Task]) -> list[PromptTask]:
    """Convert loader Tasks to PromptTasks for AgentLightning."""
    return [PromptTask(**t) for t in tasks]


def run_optimization(
    prompt_template: str,
    dataset_path: Path,
    config: APOConfig,
) -> tuple[str, float, float]:
    """Run APO optimization and return (best_prompt, initial_score, best_score)."""

    # Load and split dataset
    tasks = load_dataset(dataset_path)
    train_tasks, val_tasks = split_dataset(tasks)

    console.print(f"\n[bold]APO Prompt Optimizer[/bold]")
    console.print(f"{'Model:':<14} {config.model}")
    console.print(f"{'Optimizer:':<14} {config.optimizer_model}")
    console.print(f"{'Dataset:':<14} {len(tasks)} tasks ({len(train_tasks)} train / {len(val_tasks)} val)")
    console.print(f"{'Eval mode:':<14} {config.eval_mode}")
    console.print(f"{'Beam:':<14} width={config.beam_width}, rounds={config.beam_rounds}")
    console.print()

    # Configure scorer and rollout
    scorer = get_scorer(config.eval_mode, config.model)
    configure_rollout(scorer, config.model)

    # Build APO algorithm
    client = _build_openai_client(config)
    algo = APO[PromptTask](
        client,
        gradient_model=config.optimizer_model,
        apply_edit_model=config.optimizer_model,
        val_batch_size=config.val_batch_size,
        gradient_batch_size=config.gradient_batch_size,
        beam_width=config.beam_width,
        branch_factor=config.branch_factor,
        beam_rounds=config.beam_rounds,
    )

    # Build trainer
    trainer = Trainer(
        algorithm=algo,
        n_runners=config.n_runners,
        initial_resources={
            "prompt_template": PromptTemplate(
                template=prompt_template,
                engine="f-string",
            )
        },
        adapter=TraceToMessages(),
    )

    # Run optimization
    console.print("[bold]Starting optimization...[/bold]\n")
    trainer.fit(
        agent=prompt_rollout,
        train_dataset=_tasks_to_prompt_tasks(train_tasks),
        val_dataset=_tasks_to_prompt_tasks(val_tasks),
    )

    # Get best prompt
    best_prompt = algo.get_best_prompt()
    best_prompt_text = best_prompt.template if hasattr(best_prompt, "template") else str(best_prompt)

    return best_prompt_text, 0.0, 0.0


def run_evaluation(
    prompt_template: str,
    dataset_path: Path,
    config: APOConfig,
) -> float:
    """Evaluate a prompt on the dataset without optimizing."""
    tasks = load_dataset(dataset_path)
    scorer = get_scorer(config.eval_mode, config.model)

    from apo.core.rollout import _run_async, _run_adk_agent

    total_score = 0.0
    for task in tasks:
        prompt_text = prompt_template.replace("{input}", task["input"])
        output = _run_async(_run_adk_agent(prompt_text, task["input"], config.model))
        score = scorer.score(prompt_text, output, task)
        total_score += score
        if config.verbose:
            console.print(f"  Input: {task['input'][:50]}... Score: {score:.2f}")

    avg_score = total_score / len(tasks)
    return avg_score
