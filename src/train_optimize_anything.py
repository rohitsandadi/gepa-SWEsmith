"""
Training script using GEPA optimize_anything API.

This is the recommended API - cleaner, more maintainable than the original.
Uses structured GEPAConfig and FitnessFn protocol instead of GEPAAdapter.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Suppress verbose LiteLLM logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

from datasets import load_dataset
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig, TrackingConfig
from gepa.utils.stop_condition import TimeoutStopCondition
from src.swe_fitness_fn import create_swe_fitness_fn
from src.cost_tracker import reset_tracker
from src.experiment_logger import ExperimentLogger, set_logger
import re
import litellm


def create_logging_proposer(run_dir: Path, reflection_lm: str):
    """Create a proposer that logs all inputs/outputs to separate files."""
    
    proposer_dir = run_dir / "proposer_calls"
    proposer_dir.mkdir(parents=True, exist_ok=True)
    call_counter = [0]  # Mutable counter
    
    # Custom prompt for learning terminal and repo-specific skills
    prompt_template = """You are helping an AI coding agent learn skills for fixing bugs in a software repository.

The agent operates inside a Docker container with:
- The repository cloned at /testbed
- Access to bash commands (grep, find, cat, sed, python, pytest, git, etc.)
- The ability to read, edit, and create files

## Current Skills
The agent currently has these learned skills:
```
{curr_skills}
```

## Evaluation Results
Below are results from the agent attempting to fix bugs using the current skills. Each entry shows:
- The problem statement (bug description)
- The agent's actions and reasoning
- Whether the fix succeeded or failed
- Test output and error messages

```
{evaluation_data}
```

## Your Task
Analyze the evaluation results and propose IMPROVED SKILLS that will help the agent:

1. **Terminal/Bash Skills**: Common command patterns, file navigation, searching code, running tests
2. **Repository-Specific Knowledge**: Project structure, key modules, common patterns in this codebase
3. **Debugging Strategies**: How to locate bugs, understand test failures, verify fixes
4. **Code Editing Patterns**: Safe ways to modify files, handle imports, avoid syntax errors

Focus on:
- Patterns that led to SUCCESS - reinforce these
- Patterns that led to FAILURE - what should the agent do differently?
- Missing knowledge that would have helped
- Common mistakes to avoid

Write the improved skills as clear, actionable instructions. Be specific and concrete.
Keep skills concise but comprehensive. The agent will see these skills before each task.

Provide the new skills within ``` blocks."""

    def logging_proposer(
        candidate: dict,
        reflective_dataset: dict,
        components_to_update: list
    ) -> dict:
        """Custom proposer that logs inputs/outputs to separate files."""
        call_counter[0] += 1
        call_num = call_counter[0]
        
        results = {}
        
        for component in components_to_update:
            curr_value = candidate.get(component, "")
            side_info = reflective_dataset.get(component, [])
            
            # Format side info as string
            side_info_str = json.dumps(side_info, indent=2, default=str)
            
            # Format the prompt
            prompt = prompt_template.format(
                curr_skills=curr_value if curr_value else "(No skills learned yet)",
                evaluation_data=side_info_str
            )
            
            # Prepare log entry
            log_entry = {
                "call_num": call_num,
                "component": component,
                "current_skills": curr_value,
                "evaluation_data": side_info,
                "full_prompt": prompt,
            }
            
            # Call the reflection LLM
            try:
                response = litellm.completion(
                    model=reflection_lm,
                    messages=[{"role": "user", "content": prompt}],
                    drop_params=True,  # Let litellm drop unsupported params like temperature for gpt-5
                )
                response_text = response.choices[0].message.content
                
                # Extract new value from ``` blocks
                match = re.search(r'```(?:\w*\n)?(.*?)```', response_text, re.DOTALL)
                new_value = match.group(1).strip() if match else response_text.strip()
                
                log_entry["llm_response"] = response_text
                log_entry["new_skills"] = new_value
                log_entry["success"] = True
                
                results[component] = new_value
                
            except Exception as e:
                log_entry["error"] = str(e)
                log_entry["success"] = False
                results[component] = curr_value  # Keep current on error
            
            # Write to separate file for this call
            call_file = proposer_dir / f"call_{call_num:03d}.json"
            with open(call_file, "w") as f:
                json.dump(log_entry, f, indent=2, default=str)
            
            # Also print summary
            print(f"\n[PROPOSER] Call {call_num} for '{component}':")
            print(f"  Current skills: {len(curr_value)} chars")
            print(f"  Evaluation items: {len(side_info)}")
            print(f"  New skills: {len(results.get(component, ''))} chars")
            print(f"  Saved to: {call_file}")
        
        return results
    
    return logging_proposer

def load_split_data(split_name="train", limit=None):
    """
    Load pre-split dataset from data/ directory.
    If split doesn't exist, will fall back to direct HuggingFace load.
    """
    data_file = Path("data") / f"pygments_{split_name}.json"

    if data_file.exists():
        print(f"Loading {split_name} split from {data_file}...")
        with open(data_file) as f:
            data = json.load(f)

        if limit:
            data = data[:limit]

        print(f"Loaded {len(data)} tasks from {split_name} split")
        return data
    else:
        print(f"WARNING: Pre-split data not found at {data_file}")
        print(f"Falling back to direct HuggingFace load. Run 'python split_dataset.py' to create splits.")
        return load_pygments_data_direct(limit=limit or 10)

def load_pygments_data_direct(limit=10):
    """Fallback: Load from HuggingFace if pre-split data not available."""
    print(f"Loading {limit} Pygments examples from SWE-smith...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")

    data = []
    count = 0
    target_repo = "swesmith/pygments__pygments"

    for item in ds:
        repo_name = item.get('repo', '')
        if target_repo in repo_name:
            data.append(item)
            count += 1
            if count >= limit:
                break
    return data

def main():
    parser = argparse.ArgumentParser(
        description="GEPA optimization using optimize_anything API (recommended)"
    )
    parser.add_argument("--train-size", type=int, default=200, help="Number of training examples")
    parser.add_argument("--val-size", type=int, default=50, help="Number of validation examples for Pareto selection")
    parser.add_argument("--model", type=str, default="gpt-5.2",
                        help="Agent model for running tasks (default: gpt-5.2)")
    parser.add_argument("--reflection-model", type=str, default="gpt-5.2-pro",
                        help="Model for GEPA reflection/prompt optimization (default: gpt-5.2-pro)")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--use-split", action="store_true", help="Use pre-split train data from data/ directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--timeout", type=int, default=43200, help="Max seconds to run (default: 12 hours)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers (Docker containers)")
    parser.add_argument("--max-metric-calls", type=int, default=600, help="Max rollouts for GEPA (controls budget)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--wandb-project", type=str, default="gepa-swesmith", help="Wandb project name")
    args = parser.parse_args()

    if args.smoke_test:
        args.train_size = 2
        args.val_size = 2
        args.max_metric_calls = 20
        print("SMOKE TEST MODE")

    # Initialize experiment logger FIRST to get run_id
    exp_logger = ExperimentLogger(log_dir="gepa_results/logs")
    set_logger(exp_logger)
    run_dir = exp_logger.log_dir

    # Setup Python logging
    log_file = run_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Reset cost tracking
    reset_tracker()

    print("\n" + "="*70)
    print("GEPA + SWE-smith Training (optimize_anything API)")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Reflection Model: {args.reflection_model}")
    print(f"Workers: {args.workers} (Docker containers)")
    print(f"Run directory: {run_dir}")
    print("="*70 + "\n")

    # 1. Load Data
    if args.use_split:
        train_data = load_split_data("train", limit=args.train_size)
        val_data = load_split_data("val", limit=args.val_size)
    else:
        data = load_pygments_data_direct(limit=args.train_size + args.val_size)
        train_data = data[:args.train_size]
        val_data = data[args.train_size:args.train_size + args.val_size]
        
        if len(val_data) < args.val_size:
            val_data = train_data[:min(args.val_size, len(train_data))]

    print(f"Loaded {len(train_data)} training, {len(val_data)} validation examples.")
    logger.info(f"Training data size: {len(train_data)}, Validation size: {len(val_data)}")

    # Determine reflection model
    reflection_model = args.reflection_model or args.model

    # Save experiment config
    exp_logger.save_config({
        "api": "optimize_anything",
        "model": args.model,
        "reflection_model": reflection_model,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "max_metric_calls": args.max_metric_calls,
        "workers": args.workers,
        "seed": args.seed,
        "timeout": args.timeout,
        "use_split": args.use_split,
        "execution_mode": "docker",
        "wandb": args.wandb,
        "wandb_project": args.wandb_project if args.wandb else None,
    })

    # 2. Create Fitness Function
    print(f"\nCreating fitness function with {args.workers} workers...")
    fitness_fn = create_swe_fitness_fn(model_name=args.model, n_workers=args.workers)
    logger.info(f"Model: {args.model}")
    logger.info(f"Execution: Docker containers via SWE-smith")

    # 3. Initial Candidate (Baseline Prompt)
    # Start with empty skills - GEPA will learn and evolve them
    # The system_template in mini.yaml already has base instructions,
    # this just populates the {{ skills }} placeholder
    initial_skills = ""
    
    seed_candidate = {"skills": initial_skills}

    # 4. Setup stop conditions
    stop_callbacks = [
        TimeoutStopCondition(timeout_seconds=args.timeout),
    ]

    # 5. Configure GEPA with structured config
    wandb_kwargs = {
        "project": args.wandb_project,
        "name": run_dir.name,
        "config": {
            "model": args.model,
            "reflection_model": reflection_model,
            "train_size": len(train_data),
            "val_size": len(val_data),
        }
    } if args.wandb else None

    # Configure LiteLLM for reflection model (OpenAI regional endpoint)
    import os
    if "openai" in reflection_model.lower() or reflection_model.startswith("gpt-"):
        os.environ["OPENAI_API_BASE"] = "https://us.api.openai.com/v1"

    # Create logging proposer to capture all proposer calls
    logging_proposer = create_logging_proposer(run_dir, reflection_model)
    
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            seed=args.seed,
            display_progress_bar=True,
            max_metric_calls=args.max_metric_calls,
            candidate_selection_strategy="pareto",
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_model,
            reflection_minibatch_size=3,
            skip_perfect_score=True,
            perfect_score=1.0,
            custom_candidate_proposer=logging_proposer,
        ),
        tracking=TrackingConfig(
            use_wandb=args.wandb,
            wandb_init_kwargs=wandb_kwargs,
        ),
        stop_callbacks=stop_callbacks,
    )

    # 6. Run GEPA
    print("\n" + "="*70)
    print("Starting GEPA Optimization (optimize_anything API)...")
    print("="*70 + "\n")

    result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,
        dataset=train_data,
        valset=val_data,
        config=config,
    )

    # 7. Results
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    
    # best_candidate is a dict: {"skills": "..."}
    best_candidate_dict = result.best_candidate
    if isinstance(best_candidate_dict, dict):
        # It's already a dict with the prompt
        best_skills = best_candidate_dict.get("skills", str(best_candidate_dict))
    else:
        # It's an object with .candidate attribute
        best_skills = best_candidate_dict.candidate["skills"]
    
    # Get best score from history
    best_score = max([item.score for item in result.history]) if result.history else 0.0
    
    print(f"\nBest Prompt (Score: {best_score:.2%}):")
    print("-" * 70)
    print(best_skills)
    print("-" * 70)
    
    # Save best skills
    skills_file = run_dir / "prompts" / "best_skills.txt"
    skills_file.parent.mkdir(exist_ok=True)
    with open(skills_file, "w") as f:
        f.write(best_skills)
    
    print(f"\nBest skills saved to: {skills_file}")
    print(f"Full results in: {run_dir}")
    
    # Summary
    exp_logger.save_summary({
        "best_score": float(best_score),
        "total_iterations": len(result.history),
        "final_skills_length": len(best_skills),
    })
    
    print(f"\nSummary:")
    print(f"  Best Score: {best_score:.2%}")
    print(f"  Total Iterations: {len(result.history)}")
    print(f"  Final Skills Length: {len(best_skills)} chars")
    
    print("\n" + "="*70)
    print("Done! 🎉")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

