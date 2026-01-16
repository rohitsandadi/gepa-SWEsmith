# GEPA + SWE-smith Integration

This project applies GEPA to the software engineering domain using SWE-smith's synthetic bug dataset. The goal: automatically discover system prompts that improve agent bug-fixing performance on Pygments.

## Architecture

### Harness (`src/swe_harness.py`)

The execution harness manages the git workspace and interfaces with mini-swe-agent:

1. Checks out `base_commit`, applies the synthetic `patch` (introducing the bug)
2. Invokes `DefaultAgent` with `LitellmModel` and `LocalEnvironment`
3. Captures the agent's patch via `git diff` and full conversation trace
4. Runs verification via pytest

The GEPA-optimized prompt is prepended to the problem statement rather than replacing mini-swe-agent's built-in templates. This preserves the agent's tool-use formatting while allowing GEPA to inject domain guidance.

### Adapter (`src/adapters/swe_adapter.py`)

Implements GEPA's adapter interface:

- **`evaluate()`**: Runs the agent on a batch of tasks, verifies against `FAIL_TO_PASS` tests, then checks `PASS_TO_PASS` for regression. Returns binary scores.
- **`make_reflective_dataset()`**: Packages agent traces + test output for the reflection LLM. Truncates to avoid token limits while preserving diagnostic signal.

### Data Pipeline

SWE-smith task instances include:
- `problem_statement`: Bug description
- `patch`: Diff that introduces the synthetic bug
- `FAIL_TO_PASS`: Tests that should pass after fix
- `PASS_TO_PASS`: Regression tests (sampled subset)

Data is pre-split into train/val/test in `data/`. The validation set is critical for Pareto selection.

## Implementation Notes

**References:**
- mini-swe-agent's programmatic API (cookbook pattern with `DefaultAgent`)
- SWE-smith harness documentation for test verification structure
- GEPA's terminal-bench example for the two-template pattern

**Agent configuration:**
- Loads `mini.yaml` config, filters to supported fields only
- `step_limit=30` to bound per-task cost
- Trace capture includes both reasoning chain and environment feedback

## Usage

```bash
# Full optimization
python src/train.py --use-split --generations 3 \
    --model openai/gpt-5-mini \
    --reflection-model openai/gpt-5.2

# Evaluation
python src/evaluate_prompts.py --split test --limit 20
```

## Results

See `example/` for a complete run. The baseline prompt (6 lines, generic instructions) evolved into a 35-line domain-specific prompt with:
- MacOS environment constraints (`sed -i ''`, no `rg`)
- `pytest-timeout` handling
- Pygments-specific heuristics (lexer inheritance, `analyse_text` patterns)
- Explicit workflow steps

Validation accuracy improved from 0% → 66.7% over 3 generations.
