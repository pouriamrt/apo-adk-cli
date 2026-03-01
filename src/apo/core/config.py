from __future__ import annotations

from dataclasses import dataclass


@dataclass
class APOConfig:
    """Configuration for the APO optimization run."""

    # Model for running rollouts (LiteLLM format: "gemini/gemini-2.5-flash", "openai/gpt-4o")
    model: str = "gemini/gemini-2.5-flash"

    # Model for APO gradient/edit (uses Gemini OpenAI-compatible API by default)
    optimizer_model: str = "gemini-2.5-flash"

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

    # Vertex AI mode (uses ADC instead of API key for Google models)
    use_vertex_ai: bool = False

    # Verbose output
    verbose: bool = False
