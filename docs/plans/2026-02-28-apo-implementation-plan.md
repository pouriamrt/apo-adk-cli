# APO Prompt Optimizer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that optimizes LLM prompts using AgentLightning's APO algorithm with Google ADK + LiteLLM for model flexibility.

**Architecture:** AgentLightning APO drives the optimization loop (beam search over textual gradients). The `@rollout` function uses Google ADK's `LlmAgent` + `InMemoryRunner` with `LiteLlm` model to execute prompts. Scoring uses either reference matching or LLM-as-judge. Gemini's OpenAI-compatible API provides the `AsyncOpenAI` client that APO needs for gradient/edit models.

**Tech Stack:** Python 3.13, agentlightning[apo], google-adk, litellm, click, rich, openai (for AsyncOpenAI client)

---

### Task 1: Project Setup — pyproject.toml and Package Skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/apo/__init__.py`
- Create: `src/apo/cli/__init__.py`
- Create: `src/apo/core/__init__.py`
- Create: `src/apo/evaluation/__init__.py`
- Create: `src/apo/data/__init__.py`
- Create: `.env.example`

**Step 1: Update pyproject.toml with dependencies and package config**

```toml
[project]
name = "ai-prompt-optimizer"
version = "0.1.0"
description = "Automatic Prompt Optimization using APO algorithm"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "agentlightning[apo]",
    "google-adk",
    "litellm",
    "openai",
    "click",
    "rich",
    "python-dotenv",
]

[project.scripts]
apo = "apo.cli.commands:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/apo"]
```

**Step 2: Create package skeleton**

Create all `__init__.py` files as empty files. Create `.env.example`:

```
# Required: at least one of these
GOOGLE_API_KEY=your-google-api-key-here
# OPENAI_API_KEY=your-openai-api-key-here
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

**Step 3: Install dependencies**

Run: `uv sync` (or `pip install -e .`)

**Step 4: Verify installation**

Run: `python -c "import apo; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add pyproject.toml src/ .env.example
git commit -m "chore: project skeleton with dependencies"
```

---

### Task 2: Dataset Loader — `src/apo/data/loader.py`

**Files:**
- Create: `src/apo/data/loader.py`
- Create: `tests/test_loader.py`
- Create: `examples/sample_dataset.json`

**Step 1: Create sample dataset**

```json
[
  {"input": "What is the capital of France?", "expected_output": "Paris"},
  {"input": "What is 2 + 2?", "expected_output": "4"},
  {"input": "Translate 'hello' to Spanish", "expected_output": "hola"},
  {"input": "What color is the sky?", "expected_output": "blue"},
  {"input": "Name a mammal", "expected_output": "dog"},
  {"input": "What is the boiling point of water in Celsius?", "expected_output": "100"},
  {"input": "Who wrote Romeo and Juliet?", "expected_output": "Shakespeare"},
  {"input": "What planet is closest to the sun?", "expected_output": "Mercury"}
]
```

**Step 2: Write failing test**

```python
# tests/test_loader.py
import json
import tempfile
from pathlib import Path

from apo.data.loader import load_dataset, DatasetError


def test_load_json_dataset():
    data = [
        {"input": "hello", "expected_output": "world"},
        {"input": "foo"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    tasks = load_dataset(Path(path))
    assert len(tasks) == 2
    assert tasks[0]["input"] == "hello"
    assert tasks[0]["expected_output"] == "world"
    assert tasks[1]["input"] == "foo"
    assert tasks[1].get("expected_output") is None


def test_load_csv_dataset():
    csv_content = "input,expected_output\nhello,world\nfoo,bar\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    tasks = load_dataset(Path(path))
    assert len(tasks) == 2
    assert tasks[0]["input"] == "hello"


def test_load_dataset_missing_input_field():
    data = [{"text": "hello"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    try:
        load_dataset(Path(path))
        assert False, "Should have raised DatasetError"
    except DatasetError:
        pass


def test_load_dataset_empty():
    data = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    try:
        load_dataset(Path(path))
        assert False, "Should have raised DatasetError"
    except DatasetError:
        pass
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_loader.py -v`
Expected: FAIL (module not found)

**Step 4: Implement loader**

```python
# src/apo/data/loader.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict


class Task(TypedDict, total=False):
    input: str
    expected_output: str


class DatasetError(Exception):
    pass


def load_dataset(path: Path) -> list[Task]:
    """Load dataset from JSON or CSV file."""
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        tasks = _load_json(path)
    elif suffix == ".csv":
        tasks = _load_csv(path)
    else:
        raise DatasetError(f"Unsupported file format: {suffix}. Use .json or .csv")

    if not tasks:
        raise DatasetError("Dataset is empty")

    for i, task in enumerate(tasks):
        if "input" not in task:
            raise DatasetError(f"Task {i} missing required 'input' field")

    return tasks


def _load_json(path: Path) -> list[Task]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise DatasetError("JSON dataset must be an array of objects")
    return data


def _load_csv(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task: Task = {"input": row["input"]}
            if "expected_output" in row and row["expected_output"]:
                task["expected_output"] = row["expected_output"]
            tasks.append(task)
    return tasks


def split_dataset(tasks: list[Task], val_ratio: float = 0.3) -> tuple[list[Task], list[Task]]:
    """Split dataset into train and validation sets."""
    split_idx = max(1, int(len(tasks) * (1 - val_ratio)))
    return tasks[:split_idx], tasks[split_idx:]
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_loader.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/apo/data/ tests/test_loader.py examples/sample_dataset.json
git commit -m "feat: dataset loader with JSON/CSV support"
```

---

### Task 3: Config — `src/apo/core/config.py`

**Files:**
- Create: `src/apo/core/config.py`

**Step 1: Implement config dataclass**

```python
# src/apo/core/config.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class APOConfig:
    """Configuration for the APO optimization run."""

    # Model for running rollouts (LiteLLM format: "gemini/gemini-2.0-flash", "openai/gpt-4o")
    model: str = "gemini/gemini-2.0-flash"

    # Model for APO gradient/edit (uses Gemini OpenAI-compatible API by default)
    optimizer_model: str = "gemini-2.0-flash"

    # APO beam search parameters
    beam_width: int = 3
    branch_factor: int = 2
    beam_rounds: int = 5
    gradient_batch_size: int = 4
    val_batch_size: int = 8

    # Parallel runners
    n_runners: int = 4

    # Evaluation mode: "auto", "reference", "llm-judge"
    eval_mode: str = "auto"

    # Verbose output
    verbose: bool = False
```

**Step 2: Verify import**

Run: `python -c "from apo.core.config import APOConfig; print(APOConfig())"`
Expected: Prints the default config

**Step 3: Commit**

```bash
git add src/apo/core/config.py
git commit -m "feat: APO configuration dataclass"
```

---

### Task 4: Scorers — `src/apo/evaluation/`

**Files:**
- Create: `src/apo/evaluation/scorer.py`
- Create: `src/apo/evaluation/reference.py`
- Create: `src/apo/evaluation/llm_judge.py`
- Create: `tests/test_scoring.py`

**Step 1: Write failing tests**

```python
# tests/test_scoring.py
from apo.evaluation.reference import reference_score
from apo.evaluation.scorer import get_scorer


def test_reference_exact_match():
    score = reference_score("Paris", "Paris")
    assert score == 1.0


def test_reference_case_insensitive():
    score = reference_score("paris", "Paris")
    assert score >= 0.9


def test_reference_partial_match():
    score = reference_score("The capital is Paris", "Paris")
    assert 0.3 < score < 1.0


def test_reference_no_match():
    score = reference_score("London", "Paris")
    assert score < 0.3


def test_get_scorer_auto_with_reference():
    scorer = get_scorer("auto")
    task = {"input": "x", "expected_output": "y"}
    # Should use reference scorer when expected_output exists
    assert scorer.mode_for_task(task) == "reference"


def test_get_scorer_auto_without_reference():
    scorer = get_scorer("auto")
    task = {"input": "x"}
    # Should use llm-judge when no expected_output
    assert scorer.mode_for_task(task) == "llm-judge"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL

**Step 3: Implement reference scorer**

```python
# src/apo/evaluation/reference.py
from __future__ import annotations

from difflib import SequenceMatcher


def reference_score(actual: str, expected: str) -> float:
    """Score actual output against expected output using fuzzy string matching."""
    actual = actual.strip()
    expected = expected.strip()

    # Exact match (case insensitive)
    if actual.lower() == expected.lower():
        return 1.0

    # Check if expected is contained in actual
    if expected.lower() in actual.lower():
        # Reward containment but penalize extra text
        ratio = len(expected) / max(len(actual), 1)
        return max(0.5, ratio)

    # Fuzzy string similarity
    return SequenceMatcher(None, actual.lower(), expected.lower()).ratio()
```

**Step 4: Implement LLM judge scorer**

```python
# src/apo/evaluation/llm_judge.py
from __future__ import annotations

import litellm


def llm_judge_score(prompt: str, output: str, model: str) -> float:
    """Score output quality using an LLM as judge. Returns 0.0 to 1.0."""
    judge_prompt = (
        "You are a strict quality judge. Score the following output on a 0.0 to 1.0 scale.\n"
        "Consider: relevance, accuracy, completeness, and coherence.\n"
        "Respond with ONLY a number between 0.0 and 1.0.\n\n"
        f"Original prompt: {prompt}\n\n"
        f"Output to judge:\n{output}"
    )

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()

    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        # Try to extract a number from the response
        import re
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5  # Default if parsing fails
```

**Step 5: Implement scorer dispatcher**

```python
# src/apo/evaluation/scorer.py
from __future__ import annotations

from apo.data.loader import Task
from apo.evaluation.reference import reference_score
from apo.evaluation.llm_judge import llm_judge_score


class Scorer:
    def __init__(self, eval_mode: str, model: str = "gemini/gemini-2.0-flash"):
        self.eval_mode = eval_mode
        self.model = model

    def mode_for_task(self, task: Task) -> str:
        """Determine scoring mode for a specific task."""
        if self.eval_mode == "reference":
            return "reference"
        if self.eval_mode == "llm-judge":
            return "llm-judge"
        # auto mode
        if task.get("expected_output"):
            return "reference"
        return "llm-judge"

    def score(self, prompt: str, output: str, task: Task) -> float:
        """Score the output for a given task."""
        mode = self.mode_for_task(task)
        if mode == "reference":
            expected = task.get("expected_output", "")
            return reference_score(output, expected)
        return llm_judge_score(prompt, output, self.model)


def get_scorer(eval_mode: str, model: str = "gemini/gemini-2.0-flash") -> Scorer:
    return Scorer(eval_mode, model)
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/apo/evaluation/ tests/test_scoring.py
git commit -m "feat: scoring system with reference matching and LLM-as-judge"
```

---

### Task 5: Rollout Function — `src/apo/core/rollout.py`

**Files:**
- Create: `src/apo/core/rollout.py`

This is where Google ADK meets AgentLightning. The `@rollout` function:
1. Receives a `PromptTemplate` from APO (the prompt being optimized)
2. Creates a Google ADK `LlmAgent` with the prompt as instruction
3. Runs it via `InMemoryRunner` with the task input
4. Scores the output and returns the reward

**Step 1: Implement rollout**

```python
# src/apo/core/rollout.py
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, TypedDict

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from agentlightning import PromptTemplate
from agentlightning.litagent import rollout

from apo.evaluation.scorer import Scorer


# Module-level scorer and model, set by optimizer before training
_scorer: Scorer | None = None
_model: str = "gemini/gemini-2.0-flash"


def configure_rollout(scorer: Scorer, model: str) -> None:
    """Configure the rollout function with scorer and model. Called before training."""
    global _scorer, _model
    _scorer = scorer
    _model = model


class PromptTask(TypedDict, total=False):
    input: str
    expected_output: str


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync code (handles nested event loops)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_queue: queue.Queue[Any] = queue.Queue()

    def worker():
        try:
            result = asyncio.run(coro)
            result_queue.put((True, result))
        except BaseException as e:
            result_queue.put((False, e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    ok, payload = result_queue.get()
    t.join()
    if ok:
        return payload
    raise payload


async def _run_adk_agent(prompt_text: str, user_input: str, model: str) -> str:
    """Create and run a Google ADK agent with the given prompt and input."""
    agent = LlmAgent(
        model=LiteLlm(model=model),
        name="prompt_eval_agent",
        instruction=prompt_text,
    )

    runner = InMemoryRunner(agent=agent, app_name="apo_optimizer")

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_input)],
    )

    response_text = ""
    async for event in runner.run_async(
        user_id="apo_user",
        session_id="apo_session",
        new_message=user_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text or ""

    return response_text


@rollout
def prompt_rollout(task: PromptTask, prompt_template: PromptTemplate) -> float:
    """Execute a single rollout: run the prompt on the task input and score the output."""
    assert _scorer is not None, "Scorer not configured. Call configure_rollout() first."

    # Format the prompt template with the task input
    prompt_text = prompt_template.format(input=task["input"])

    # Run ADK agent
    output = _run_async(_run_adk_agent(prompt_text, task["input"], _model))

    # Score the output
    reward = _scorer.score(prompt_text, output, task)
    return reward
```

**Step 2: Verify import**

Run: `python -c "from apo.core.rollout import prompt_rollout, configure_rollout; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/apo/core/rollout.py
git commit -m "feat: rollout function bridging ADK agent with AgentLightning APO"
```

---

### Task 6: Optimizer Orchestrator — `src/apo/core/optimizer.py`

**Files:**
- Create: `src/apo/core/optimizer.py`

**Step 1: Implement orchestrator**

```python
# src/apo/core/optimizer.py
from __future__ import annotations

import os
from pathlib import Path

from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table

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

    return best_prompt_text, 0.0, 0.0  # scores will be filled from APO logs


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
```

**Step 2: Verify import**

Run: `python -c "from apo.core.optimizer import run_optimization; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/apo/core/optimizer.py
git commit -m "feat: optimizer orchestrator connecting APO trainer with ADK rollouts"
```

---

### Task 7: CLI Commands — `src/apo/cli/commands.py`

**Files:**
- Create: `src/apo/cli/commands.py`
- Create: `examples/sample_prompt.txt`

**Step 1: Create sample prompt**

```text
Answer the following question accurately and concisely.

{input}
```

Save to `examples/sample_prompt.txt`.

**Step 2: Implement CLI**

```python
# src/apo/cli/commands.py
from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from apo.core.config import APOConfig

console = Console()


@click.group()
def cli():
    """APO — Automatic Prompt Optimizer"""
    load_dotenv()


@cli.command()
@click.option("--prompt", type=str, help="Prompt template string (must contain {input})")
@click.option("--prompt-file", type=click.Path(exists=True), help="Path to prompt template file")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to dataset (JSON/CSV)")
@click.option("--model", default="gemini/gemini-2.0-flash", help="LiteLLM model string for rollouts")
@click.option("--optimizer-model", default=None, help="Model for APO gradient/edit (defaults to gemini-2.0-flash)")
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
    model: str,
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

    config = APOConfig(
        model=model,
        optimizer_model=optimizer_model or "gemini-2.0-flash",
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
@click.option("--model", default="gemini/gemini-2.0-flash", help="LiteLLM model string")
@click.option("--eval-mode", type=click.Choice(["auto", "reference", "llm-judge"]), default="auto")
@click.option("--verbose", "-v", is_flag=True)
def evaluate(
    prompt: str | None,
    prompt_file: str | None,
    dataset: str,
    model: str,
    eval_mode: str,
    verbose: bool,
):
    """Evaluate a prompt on a dataset (no optimization)."""
    prompt_text = _load_prompt(prompt, prompt_file)

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
```

**Step 3: Verify CLI help**

Run: `python -m apo.cli.commands --help`
Expected: Shows CLI help with `optimize` and `evaluate` commands

**Step 4: Commit**

```bash
git add src/apo/cli/ examples/sample_prompt.txt
git commit -m "feat: CLI with optimize and evaluate commands"
```

---

### Task 8: Integration Test — End-to-End Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""
Integration test - requires GOOGLE_API_KEY or OPENAI_API_KEY set.
Run with: pytest tests/test_integration.py -v -s
"""
import os
import pytest
from pathlib import Path
from click.testing import CliRunner

from apo.cli.commands import cli


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="No API key available",
)
class TestIntegration:
    def test_evaluate_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate",
            "--prompt", "Answer concisely: {input}",
            "--dataset", str(Path(__file__).parent.parent / "examples" / "sample_dataset.json"),
            "--model", "gemini/gemini-2.0-flash",
            "--eval-mode", "reference",
        ])
        assert result.exit_code == 0
        assert "Average score" in result.output

    def test_optimize_minimal(self):
        """Minimal optimization with small beam to verify the pipeline works."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "optimize",
            "--prompt", "Answer: {input}",
            "--dataset", str(Path(__file__).parent.parent / "examples" / "sample_dataset.json"),
            "--model", "gemini/gemini-2.0-flash",
            "--beam-width", "2",
            "--beam-rounds", "1",
            "--n-runners", "2",
            "--eval-mode", "reference",
        ])
        assert result.exit_code == 0
        assert "Optimization complete" in result.output
```

**Step 2: Run integration test (if API key available)**

Run: `python -m pytest tests/test_integration.py -v -s`
Expected: PASS (or SKIP if no API key)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration tests for CLI"
```

---

### Task 9: Main Entry Point and Final Polish

**Files:**
- Modify: `main.py`
- Modify: `src/apo/__init__.py`

**Step 1: Update main.py**

```python
# main.py
from apo.cli.commands import cli

if __name__ == "__main__":
    cli()
```

**Step 2: Update src/apo/__init__.py**

```python
# src/apo/__init__.py
"""APO — Automatic Prompt Optimizer using AgentLightning + Google ADK."""

__version__ = "0.1.0"
```

**Step 3: Verify end-to-end CLI**

Run: `apo --help`
Expected: Shows CLI help

Run: `apo optimize --help`
Expected: Shows optimize command help with all options

**Step 4: Final commit**

```bash
git add main.py src/apo/__init__.py
git commit -m "feat: wire up main entry point and package metadata"
```

---

### Task 10: Manual End-to-End Test

**No files to create. This is a manual verification step.**

**Step 1: Set up environment**

Create `.env` file from `.env.example` with a real API key.

**Step 2: Run evaluate**

```bash
apo evaluate \
  --prompt-file examples/sample_prompt.txt \
  --dataset examples/sample_dataset.json \
  --model "gemini/gemini-2.0-flash" \
  --eval-mode reference \
  -v
```

Expected: Prints per-task scores and average score.

**Step 3: Run optimize (small)**

```bash
apo optimize \
  --prompt-file examples/sample_prompt.txt \
  --dataset examples/sample_dataset.json \
  --model "gemini/gemini-2.0-flash" \
  --beam-width 2 \
  --beam-rounds 2 \
  --n-runners 2 \
  --output optimized_prompt.txt \
  -v
```

Expected: Shows optimization progress, prints best prompt, saves to file.

**Step 4: Verify the optimized prompt is better**

```bash
apo evaluate \
  --prompt-file optimized_prompt.txt \
  --dataset examples/sample_dataset.json \
  --model "gemini/gemini-2.0-flash" \
  --eval-mode reference
```

Expected: Score should be equal or higher than the original.

---

## Dependency Install Order

1. `uv sync` or `pip install -e ".[dev]"` (installs all deps)
2. Verify: `python -c "import agentlightning; import google.adk; import litellm; print('All deps OK')"`

## Key Reference Docs

- AgentLightning APO API: https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/apo/
- AgentLightning `@rollout`: https://microsoft.github.io/agent-lightning/latest/tutorials/write-agents/
- APO example code: https://github.com/microsoft/agent-lightning/blob/main/examples/apo/room_selector_apo.py
- Google ADK + LiteLLM: https://google.github.io/adk-docs/agents/models/litellm/
- ADK InMemoryRunner: https://google.github.io/adk-docs/runtime/
- Gemini OpenAI-compatible API: https://ai.google.dev/gemini-api/docs/openai
