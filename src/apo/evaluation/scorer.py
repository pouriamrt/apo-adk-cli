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
