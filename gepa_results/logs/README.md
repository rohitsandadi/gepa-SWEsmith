# GEPA Experiment Logs

## Folder Structure

```
gepa_results/logs/
├── latest/                    # Symlink to most recent run
└── run_YYYYMMDD_HHMMSS_xxx/   # One folder per experiment run
    │
    ├── config.json            # Experiment settings (model, sizes, etc.)
    │
    ├── summary.json           # MAIN RESULTS: before/after comparison
    │                          # - Baseline vs final avg tokens
    │                          # - Baseline vs final avg steps
    │                          # - Pass rate improvement
    │
    ├── iterations.jsonl       # Per-evaluation metrics
    │                          # - Each line = one batch of tasks evaluated
    │                          # - Shows tokens/steps for each task
    │
    ├── proposer_inputs.jsonl  # What the proposer LLM sees
    │                          # - Current prompt + failure traces
    │                          # - Use this to debug/improve proposer
    │
    ├── prompts/               # All prompts generated
    │   ├── prompt_abc123.txt  # Baseline prompt (first hash)
    │   └── prompt_def456.txt  # GEPA-generated prompts
    │
    ├── best_prompt.txt        # Final best prompt (copy for convenience)
    ├── baseline_prompt.txt    # Starting prompt (copy for convenience)
    │
    ├── training.log           # Human-readable log of the run
    └── cost_summary.txt       # API costs breakdown
```

## Where to Find What You Need

- **What it shows**:
  - Detailed per-task metrics (tokens, steps)
  - Exactly what the proposer sees (for debugging)
  - All generated prompts


| Request | Where to Find It |
|---------|------------------|
| "avg tokens before/after" | `summary.json` → comparison.baseline.avg_tokens vs comparison.latest.avg_tokens |
| "avg steps before/after" | `summary.json` → comparison.baseline.avg_steps vs comparison.latest.avg_steps |
| "prompt to proposer" | `proposer_inputs.jsonl` → current_prompt field |
| "tracess to proposer" | `proposer_inputs.jsonl` → reflection_records field |

## Quick Analysis Commands

```bash
# See before/after comparison
cat gepa_results/logs/latest/summary.json | python -m json.tool

# See what proposer received (first entry)
head -1 gepa_results/logs/latest/proposer_inputs.jsonl | python -m json.tool

# Compare baseline vs final prompt
diff gepa_results/logs/latest/baseline_prompt.txt gepa_results/logs/latest/best_prompt.txt

# Count unique prompts generated
ls gepa_results/logs/latest/prompts/prompt_*.txt | wc -l
```

## Log Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         GEPA LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. EVALUATE baseline prompt on tasks                           │
│     └─→ Logged to: iterations.jsonl (eval_id: 0)                │
│                                                                  │
│  2. COLLECT failure traces (agent reasoning + test output)      │
│     └─→ Logged to: proposer_inputs.jsonl                        │
│                                                                  │
│  3. PROPOSER generates new prompt based on failures             │
│     └─→ Logged to: prompts/prompt_<new_hash>.txt                │
│                                                                  │
│  4. EVALUATE new prompt on tasks                                │
│     └─→ Logged to: iterations.jsonl (eval_id: 1, 2, ...)        │
│                                                                  │
│  5. REPEAT until budget exhausted or no improvement             │
│                                                                  │
│  6. SAVE best prompt and summary                                │
│     └─→ best_prompt.txt, summary.json                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## File Formats

### `config.json` - Experiment Configuration

Saved at START of run. Contains all settings used.

```json
{
  "run_id": "run_20260117_125631_7f437e",
  "start_time": "2026-01-17T12:56:31.306048",
  "model": "openai/gpt-5.2",
  "reflection_model": "openai/gpt-5.2",
  "train_size": 2,
  "val_size": 2,
  "max_metric_calls": 6,
  "workers": 2,
  "seed": 42
}
```

---

### `iterations.jsonl` - Per-Evaluation Metrics

**Format**: JSON Lines (one JSON object per line)

Each line = one call to `adapter.evaluate()`.

```json
{
  "eval_id": 0,
  "timestamp": "2026-01-17T12:58:32.998595",
  "prompt_hash": "da5c5ddf",
  "prompt_preview": "You are an autonomous software engineer...",
  "num_tasks": 2,
  "num_passed": 0,
  "pass_rate": 0.0,
  "avg_steps": 18.0,
  "avg_tokens": 13101.0,
  "total_steps": 36,
  "total_tokens": 26202,
  "task_metrics": [
    {
      "instance_id": "pygments__pygments.27649ebb.combine_file__plfenvp1",
      "success": false,
      "score": 0.0,
      "steps": 22,
      "estimated_tokens": 12974,
      "has_patch": false
    }
  ]
}
```

**Fields**:
| Field | Meaning |
|-------|---------|
| `eval_id` | Which evaluation batch (0 = first/baseline) |
| `prompt_hash` | 8-char hash → find prompt in `prompts/prompt_<hash>.txt` |
| `avg_steps` | Average agent turns per task (lower = more efficient) |
| `avg_tokens` | Average tokens per task (lower = cheaper) |
| `task_metrics[]` | Per-task breakdown with instance_id, steps, tokens |

---

### `proposer_inputs.jsonl` - What Proposer LLM Sees

**Format**: JSON Lines

Each line = one call to the proposer (reflection LLM).

```json
{
  "iteration": 2,
  "timestamp": "2026-01-17T13:01:45.123456",
  "current_prompt": "You are an autonomous software engineer...",
  "reflection_records": [
    {
      "Inputs": {
        "Task ID": "pygments__pygments.27649ebb...",
        "Problem Statement": "Fix the HTML formatter..."
      },
      "Generated Outputs": {
        "Agent Reasoning & Actions": "[ASSISTANT] Let me search...",
        "Patch Generated": "diff --git a/..."
      },
      "Environment Feedback": {
        "Test Verification Output": "FAILED tests/test_html.py...",
        "Status": "FAIL_TO_PASS tests failed",
        "Score": 0.0
      }
    }
  ]
}
```

**Fields**:
| Field | Meaning |
|-------|---------|
| `current_prompt` | The prompt that led to these failures |
| `reflection_records[]` | Array of failed tasks |
| `Agent Reasoning & Actions` | What the agent tried (chain of thought) |
| `Patch Generated` | The code changes made |
| `Test Verification Output` | Why it failed (test errors, stack traces) |

**Use case**: If GEPA isn't generating good prompts, examine this to see what information the proposer is receiving.

---

### `summary.json` - Final Results

Saved at END of run.

```json
{
  "experiment_start": "2026-01-17T12:56:31",
  "experiment_end": "2026-01-17T13:05:12",
  "total_eval_batches": 3,
  "unique_prompts_tested": 2,
  "comparison": {
    "baseline": {
      "prompt_hash": "da5c5ddf",
      "pass_rate": 0.0,
      "avg_steps": 18.0,
      "avg_tokens": 13101.0
    },
    "latest": {
      "prompt_hash": "6198e314",
      "pass_rate": 0.0,
      "avg_steps": 27.7,
      "avg_tokens": 14675.0
    },
    "improvement": {
      "pass_rate_delta": 0.0,
      "steps_delta": 9.7,
      "tokens_delta": 1574.0
    }
  },
  "prompts_seen": ["da5c5ddf", "6198e314"]
}
```


---

### `prompts/prompt_<hash>.txt` - Prompt Files

**Format**: Plain text

One file per unique prompt. Filename = `prompt_` + 8-char hash + `.txt`

To find a prompt:
1. Get `prompt_hash` from `iterations.jsonl` or `summary.json`
2. Open `prompts/prompt_<hash>.txt`

Example:
```
prompt_hash: "da5c5ddf" → prompts/prompt_da5c5ddf.txt
prompt_hash: "6198e314" → prompts/prompt_6198e314.txt
```
