# APO Prompt Optimizer — Design Document

**Date:** 2026-02-28
**Status:** Approved

## Overview

A CLI tool that automatically optimizes LLM prompts using the APO (Automatic Prompt Optimization) algorithm. Built with Microsoft AgentLightning for the APO training loop and Google ADK with LiteLLM for flexible model access.

## Architecture

```
CLI (click) → Orchestrator → AgentLightning APO Trainer
                                    │
                              @agl.rollout
                                    │
                            Google ADK LlmAgent
                            (model=LiteLlm(...))
                                    │
                              Scorer (reward)
```

### Data Flow

1. User provides: initial prompt template, dataset (JSON/CSV), model choice, eval mode
2. APO trainer samples tasks → calls rollout function → rollout formats prompt with task input → ADK agent generates output → scorer computes reward → reward returns to APO
3. APO generates textual gradients from low-scoring rollouts, edits prompt, beam-searches for best candidates
4. After N rounds, best prompt is printed/saved

## Dataset Format

JSON file with array of objects:

```json
[
  {"input": "Summarize: ...", "expected_output": "A concise summary..."},
  {"input": "Translate: Hello"},
]
```

- `input` (required): variable part of each task
- `expected_output` (optional): enables reference-based scoring

Also supports CSV with `input` and optional `expected_output` columns.

## Prompt Template Format

Text with `{input}` placeholder (required). Additional placeholders become part of what APO optimizes.

```
You are a helpful assistant. {instruction}

User input: {input}
```

## Scoring

Three modes via `--eval-mode`:

| Mode | When | How |
|------|------|-----|
| `reference` | `expected_output` present | Fuzzy match + LLM semantic similarity |
| `llm-judge` | No reference | Separate LLM grades output 0.0–1.0 |
| `auto` (default) | Mixed | Reference when available, LLM-judge otherwise |

## CLI Interface

```bash
# Optimize a prompt
apo optimize \
  --prompt "You are a helpful assistant. Respond to: {input}" \
  --dataset data.json \
  --model "gemini/gemini-2.0-flash" \
  --eval-mode auto \
  --beam-width 3 \
  --beam-rounds 5 \
  --output optimized_prompt.txt

# Load prompt from file
apo optimize --prompt-file prompt.txt --dataset data.json

# Evaluate without optimizing
apo evaluate --prompt-file prompt.txt --dataset data.json
```

## Project Structure

```
ai-prompt-optimizer/
├── pyproject.toml
├── .env.example
├── src/
│   └── apo/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── commands.py         # Click CLI (optimize, evaluate)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── optimizer.py        # Orchestrator: configures & runs APO trainer
│       │   ├── rollout.py          # @agl.rollout + ADK agent setup
│       │   └── config.py           # APO defaults, model config dataclass
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── scorer.py           # Base scorer protocol + auto-detection
│       │   ├── reference.py        # Reference-based scoring
│       │   └── llm_judge.py        # LLM-as-judge scoring
│       └── data/
│           ├── __init__.py
│           └── loader.py           # Dataset loading + validation
├── examples/
│   ├── sample_dataset.json
│   └── sample_prompt.txt
└── tests/
```

## Dependencies

- `google-adk` — agent framework + LiteLLM model integration
- `agentlightning[apo]` — APO algorithm + trainer
- `click` — CLI framework
- `rich` — terminal output (progress bars, tables)

## APO Configuration Defaults

- `beam_width=3`, `branch_factor=2`, `beam_rounds=5`
- `gradient_batch_size=4`, `val_batch_size=8`
- All configurable via CLI flags

## Error Handling

- Missing API key → clear message with setup instructions
- Invalid dataset → validation errors with line/field info
- LLM API errors → retry with exponential backoff (3 attempts)
- No improvement → report original prompt was best

## Output

- Best prompt printed to stdout
- `--output` flag saves to file
- `--verbose` for per-rollout details
- Exit code 0 success, 1 error

## Technology References

- [APO Paper (arXiv)](https://arxiv.org/abs/2305.03495)
- [AgentLightning APO Docs](https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/apo/)
- [AgentLightning GitHub](https://github.com/microsoft/agent-lightning)
- [Google ADK Docs](https://google.github.io/adk-docs/)
- [ADK + LiteLLM](https://google.github.io/adk-docs/agents/models/litellm/)
