from __future__ import annotations

import re

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
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5
