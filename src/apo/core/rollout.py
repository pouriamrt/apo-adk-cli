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
