# GEPA + SWE-smith Integration

Applying GEPA prompt optimization to software engineering tasks using SWE-smith's synthetic bug dataset. This project explores whether prompts optimized on training problems from one repository generalize to held-out test problems from the same repository.

## Overview

| Component | Role |
|-----------|------|
| **GEPA** | Genetic-Pareto prompt optimization via LLM reflection |
| **SWE-smith** | Synthetic bug dataset (Pygments subset) |
| **mini-swe-agent** | Lightweight agent for task execution |

The optimization loop: evaluate prompts on tasks → capture agent traces + test output → reflect on failures → mutate prompt → repeat.

For implementation details, see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

## Project Structure

```
├── src/
│   ├── train.py              # GEPA optimization entry point
│   ├── swe_harness.py        # Agent execution + git workspace management
│   ├── evaluate_prompts.py   # Baseline vs optimized comparison
│   └── adapters/
│       └── swe_adapter.py    # GEPA adapter (evaluate + reflective dataset)
├── example/                   # Complete run with baseline → optimized prompt
├── PROJECT_SUMMARY.md         # Implementation details
└── requirements.txt
```

## Usage

```bash
# Install
pip install -r requirements.txt

# Run optimization
python src/train.py --use-split --generations 3 \
    --model openai/gpt-5-mini \
    --reflection-model openai/gpt-5.2

# Evaluate
python src/evaluate_prompts.py --split test --limit 20
```

## Results

See `example/` for a complete run. The baseline prompt (6 lines) evolved into a 35-line domain-specific prompt including MacOS constraints, pytest-timeout handling, and Pygments-specific heuristics. Validation accuracy: 0% → 66.7%.

## References

- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [SWE-smith](https://swesmith.com)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
