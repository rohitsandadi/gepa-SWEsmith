# GEPA + SWE-smith Integration

Applying GEPA prompt optimization to software engineering tasks using SWE-smith's synthetic bug dataset. All tasks run in isolated Docker containers for perfect reproducibility.

## Overview

| Component | Role |
|-----------|------|
| **GEPA** | Genetic-Pareto prompt optimization via LLM reflection |
| **SWE-smith** | Synthetic bug dataset + Docker containers (Pygments subset) |
| **mini-swe-agent** | Lightweight agent for task execution |

**Key Features:**
- 🐳 Docker containers via SWE-smith (zero manual setup)
- ⚡ Two APIs: `optimize_anything` (recommended) and original
- 🎯 Parallel execution with perfect isolation
- 📊 Automatic cost tracking via LiteLLM

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install swesmith docker

# 2. Run training (uses optimize_anything API)
python src/train_optimize_anything.py --use-split --train-size 50 --workers 6

# 3. Evaluate results
python src/evaluate_prompts.py --split test --limit 20
```

## Docker Execution

All tasks automatically run in **SWE-smith Docker containers**:

✅ Pre-configured environment (repo cloned, bug applied, tests ready)  
✅ Perfect isolation (no workspace conflicts)  
✅ Reproducible everywhere  
✅ Scales easily with `--workers N`  

No manual setup required - Docker containers are created and cleaned up automatically.

## Two APIs Available

### Option 1: optimize_anything API (Recommended) ⭐

**Cleaner, more maintainable** - Uses simple fitness function and structured config.

```bash
python src/train_optimize_anything.py \
    --use-split \
    --train-size 100 \
    --val-size 50 \
    --model gemini/gemini-2.0-flash-exp \
    --reflection-model gemini/gemini-2.0-flash-thinking-exp-01-21 \
    --workers 6 \
    --max-metric-calls 600
```

**Key files:**
- `src/train_optimize_anything.py` - Training script
- `src/swe_fitness_fn.py` - Fitness function wrapper

### Option 2: Original API (Legacy)

**Full control** - Uses adapter protocol with more customization options.

```bash
python src/train.py \
    --use-split \
    --train-size 100 \
    --val-size 50 \
    --model gemini/gemini-2.0-flash-exp \
    --reflection-model gemini/gemini-2.0-flash-thinking-exp-01-21 \
    --workers 6 \
    --max-metric-calls 600
```

**Key files:**
- `src/train.py` - Training script
- `src/adapters/swe_adapter.py` - Adapter implementation

**Both APIs produce equivalent results** - same GEPA engine, same performance.

## Command-Line Options

```
--train-size N        Training examples (default: 200)
--val-size N          Validation examples (default: 50)
--model MODEL         Agent model (default: gemini-2.0-flash-exp)
--reflection-model M  Reflection model (default: gemini-2.0-flash-thinking-exp)
--workers N           Parallel containers (default: 6)
--max-metric-calls N  Max rollouts (default: 600)
--timeout N           Max seconds (default: 43200)
--seed N              Random seed (default: 42)
--smoke-test          Quick 2-task test
--use-split           Use pre-split data from data/
--wandb               Enable wandb tracking
--verbose             Verbose logging
```

## Examples

### Smoke Test
```bash
python src/train_optimize_anything.py --smoke-test
```

### Small Run (Development)
```bash
python src/train_optimize_anything.py --use-split --train-size 10 --workers 4
```

### Full Training (Production)
```bash
python src/train_optimize_anything.py \
    --use-split \
    --train-size 200 \
    --val-size 50 \
    --workers 8 \
    --max-metric-calls 1000 \
    --wandb \
    --wandb-project my-experiment
```

### With OpenAI Models
```bash
python src/train_optimize_anything.py \
    --use-split \
    --model openai/gpt-4o-mini \
    --reflection-model openai/gpt-4o \
    --workers 6
```

### Evaluation
```bash
python src/evaluate_prompts.py \
    --prompt-file gepa_results/logs/run_*/prompts/best_prompt.txt \
    --split test \
    --limit 50
```

## Project Structure

```
├── src/
│   ├── train_optimize_anything.py  # Training (optimize_anything API)
│   ├── train.py                    # Training (original API)
│   ├── swe_fitness_fn.py          # Fitness function wrapper
│   ├── swe_harness.py             # Docker container management
│   ├── adapters/
│   │   └── swe_adapter.py         # Original adapter
│   ├── evaluate_prompts.py        # Evaluation script
│   ├── split_dataset.py           # Dataset splitting
│   ├── cost_tracker.py            # LiteLLM cost tracking
│   └── experiment_logger.py       # Experiment logging
├── data/                           # Pre-split datasets (optional)
├── gepa_results/logs/             # Training outputs
├── requirements.txt
└── README.md
```

## Results Structure

Results are saved to `gepa_results/logs/run_YYYYMMDD_HHMMSS_<id>/`:

```
run_20260127_143022_abc123/
├── config.json              # Experiment configuration
├── iterations.jsonl         # Per-iteration logs
├── prompts/
│   └── best_prompt.txt      # Best prompt found
├── summary.json             # Final results
└── training.log             # Debug logs
```

## API Comparison

| Feature | optimize_anything | Original |
|---------|-------------------|----------|
| **Code complexity** | Simple (~150 lines) | Complex (~300 lines) |
| **Configuration** | Structured objects | Flat parameters |
| **Integration** | Single function | Adapter protocol |
| **Maintainability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Performance** | Same | Same |
| **Results** | Identical | Identical |

**Recommendation:** Use `optimize_anything` for new projects.

## How It Works

### 1. Docker Container Setup
```python
from swesmith.profiles import registry

# Get pre-configured container for task
rp = registry.get_from_inst(task)
container = rp.get_container(task)
# Container has: repo cloned, bug applied, deps installed
```

### 2. Agent Execution
```python
# Run agent in container
agent = DefaultAgent(model=..., env=LocalEnvironment(cwd="/testbed"))
agent.run(problem_statement)

# Get patch
patch = container.exec_run("git diff", workdir="/testbed")
```

### 3. Test Verification
```python
# Run tests in container
result = container.exec_run("pytest", workdir="/testbed")
passed = result.exit_code == 0
```

### 4. GEPA Optimization
```python
# optimize_anything API
result = optimize_anything(
    seed_candidate={"system_prompt": initial_prompt},
    fitness_fn=fitness_fn,
    dataset=train_data,
    valset=val_data,
    config=GEPAConfig(...)
)
```

## Requirements

### System
- Docker (rootless or standard)
  - Rootless: `systemctl --user start docker`
  - Standard: `sudo systemctl start docker`
- 8GB+ RAM (4 workers) or 16GB+ RAM (8 workers)
- Python 3.8+

### Python Packages
```bash
pip install -r requirements.txt
pip install swesmith docker
```

### Environment Variables
Create `.env` from `.env.example`:
```bash
cp .env.example .env
# Add your API keys
```

Required: At least one LLM provider API key (Google, OpenAI, Anthropic, or Groq).

## Troubleshooting

### "Cannot connect to Docker daemon"
```bash
# Rootless Docker (recommended)
systemctl --user start docker
systemctl --user enable docker  # Auto-start on boot

# If still failing, set DOCKER_HOST explicitly:
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
# Or add to your ~/.bashrc or ~/.zshrc

# Or standard Docker
sudo systemctl start docker

# Or Docker Desktop (Mac/Windows)
# Start Docker Desktop application
```

### "swesmith module not found"
```bash
pip install swesmith
# Or from source:
git clone https://github.com/SWE-bench/SWE-smith.git
cd SWE-smith && pip install -e .
```

### Out of memory
```bash
# Reduce parallel workers
python src/train_optimize_anything.py --workers 2
```

### "gepa.optimize_anything not found"
```bash
# Check gepa-optimize-anything is installed
pip install -e /home/eecs/shangyin/gepa-optimize-anything
```

## Example Results

See `example/` for a complete run. The baseline prompt (6 lines) evolved into a 35-line domain-specific prompt with:
- MacOS environment constraints
- pytest-timeout handling
- Pygments-specific heuristics
- Explicit workflow steps

**Validation accuracy: 0% → 66.7%**

## References

- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [SWE-smith](https://github.com/SWE-bench/SWE-smith) ([Paper](https://arxiv.org/abs/2504.21798))
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
- [SWE-bench](https://swe-bench.github.io/)## LicenseMIT - see LICENSE file.## Citation```bibtex
@inproceedings{yang2025swesmith,
  title={SWE-smith: Scaling Data for Software Engineering Agents}, 
  author={John Yang and Kilian Lieret and Carlos E. Jimenez and others},
  booktitle={NeurIPS 2025 D&B Spotlight},
  year={2025}
}
```