import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from datasets import load_dataset
from gepa.api import optimize
from gepa.utils.stop_condition import (
    TimeoutStopCondition,
    NoImprovementStopper,
)
from src.adapters.swe_adapter import SWEAdapter
from src.cost_tracker import get_tracker

def load_split_data(split_name="train", limit=None):
    """
    Load pre-split dataset from data/ directory.
    If split doesn't exist, will fall back to streaming load.
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
        print(f"Falling back to streaming load. Run 'python split_dataset.py' to create splits.")
        return load_pygments_data_streaming(limit=limit or 10)

def load_pygments_data_streaming(limit=10):
    """Fallback: Stream from HuggingFace if pre-split data not available."""
    print(f"Loading {limit} Pygments examples from SWE-smith (streaming)...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=2, help="Number of GEPA generations")
    parser.add_argument("--train-size", type=int, default=5, help="Number of training examples")
    parser.add_argument("--val-size", type=int, default=5, help="Number of validation examples for Pareto selection")
    parser.add_argument("--workspace", type=str, default="/tmp/gepa_workenvs/pygments")
    parser.add_argument("--model", type=str, default="gemini/gemma-3-27b-it", 
                        help="Agent model for running tasks (default: gemini/gemma-3-27b-it)")
    parser.add_argument("--reflection-model", type=str, default=None,
                        help="Model for GEPA reflection/prompt optimization (default: same as --model)")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--use-split", action="store_true", help="Use pre-split train data from data/ directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--timeout", type=int, default=3600, help="Max seconds to run (default: 1 hour)")
    parser.add_argument("--stop-if-no-improvement", type=int, default=10, help="Stop after N iterations without improvement")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Setup logging
    log_dir = Path("gepa_results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "training.log"

    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("GEPA Training Session Started")
    logger.info("="*70)

    if args.smoke_test:
        args.generations = 1
        args.train_size = 2
        print("TEST MODE")
        logger.info("Running in TEST mode: 1 generation, 2 tasks")

    # Initialize cost tracker
    tracker = get_tracker()
    print(f"\n💰 Cost tracking enabled")
    print(f"   Logs: {log_dir}")
    print(f"   Training log: {log_file}")
    print(f"   Cost summary: {log_dir / 'cost_summary.txt'}\n")

    # 1. Load Data (train and validation for Pareto selection)
    if args.use_split:
        print("\n" + "="*70)
        print("Using pre-split datasets")
        print("="*70)
        train_data = load_split_data(split_name="train", limit=args.train_size)
        val_data = load_split_data(split_name="val", limit=args.val_size)
    else:
        print("\n" + "="*70)
        print("Loading data via streaming (not using pre-split)")
        print("Recommendation: Run 'python split_dataset.py' and use --use-split flag")
        print("="*70)
        train_data = load_pygments_data_streaming(limit=args.train_size)
        # Use subset for validation when streaming
        val_data = train_data[:min(args.val_size, len(train_data))]

    print(f"Loaded {len(train_data)} training, {len(val_data)} validation examples.")
    logger.info(f"Training data size: {len(train_data)}, Validation size: {len(val_data)}")

    # 2. Setup Adapter
    adapter = SWEAdapter(workspace_root=args.workspace, model_name=args.model)
    logger.info(f"Model: {args.model}")
    logger.info(f"Workspace: {args.workspace}")

    # 3. Initial Candidate (The Baseline Prompt)
    # We use a simplified version of the standard SWE-agent prompt
    initial_prompt = """
You are an autonomous software engineer.
You will be given a specific issue to fix in a software repository.
You have access to the codebase and can write files and run commands.
Your goal is to reproduce the issue (if possible) and then fix it.
You MUST generate a patch using `git diff` implicitly by changing files.
When you are done, submit your changes.
    """.strip()
    
    seed_candidate = {"system_prompt": initial_prompt}

    # 4. Setup stop conditions
    stop_callbacks = [
        TimeoutStopCondition(timeout_seconds=args.timeout),
        NoImprovementStopper(max_iterations_without_improvement=args.stop_if_no_improvement),
    ]

    # 5. Run GEPA
    print("\n" + "="*70)
    # Determine reflection model (can be different from agent model)
    reflection_model = args.reflection_model or args.model

    print("Starting GEPA Optimization...")
    print("="*70)
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Agent model: {args.model}")
    logger.info(f"Reflection model: {reflection_model}")
    logger.info(f"Initial prompt: {initial_prompt[:100]}...")

    print(f"\n📊 Watch progress:")
    print(f"   tail -f {log_file}")
    print(f"   tail -f {log_dir / 'cost_summary.txt'}")
    print()

    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,  # Critical for Pareto selection
        adapter=adapter,
        # GEPA hyperparameters
        reflection_lm=reflection_model,  # Model for reflection (can differ from agent)
        max_metric_calls=args.generations * args.train_size * 2,  # Rough budget
        reflection_minibatch_size=3,  # GEPA default (was 1)
        candidate_selection_strategy="pareto",  # Explicit Pareto selection
        skip_perfect_score=True,  # Skip reflection on perfect scores
        perfect_score=1.0,
        use_wandb=False,
        run_dir="gepa_results",
        display_progress_bar=True,  # Nice progress bar
        seed=args.seed,  # For reproducibility
        stop_callbacks=stop_callbacks,
    )

    logger.info("GEPA optimization completed")
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print("\nBest Candidate Found:")
    print(result.best_candidate)

    # Save best prompt to file for easy evaluation
    output_dir = Path("gepa_results")
    output_dir.mkdir(exist_ok=True)

    best_prompt_file = output_dir / "best_prompt.txt"
    with open(best_prompt_file, 'w') as f:
        f.write(result.best_candidate.get("system_prompt", ""))

    print(f"\n✓ Best prompt saved to: {best_prompt_file}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on test set:")
    print(f"     python src/evaluate_prompts.py --split test --limit 20")
    print(f"  2. Compare baseline vs optimized:")
    print(f"     python src/evaluate_prompts.py --baseline baseline_prompt.txt --optimized {best_prompt_file}")
    print(f"  3. Review full results in: gepa_results/")

    # Also save baseline for comparison
    baseline_file = output_dir / "baseline_prompt.txt"
    if not baseline_file.exists():
        baseline_prompt = """You are an autonomous software engineer.
You will be given a specific issue to fix in a software repository.
You have access to the codebase and can write files and run commands.
Your goal is to reproduce the issue (if possible) and then fix it.
You MUST generate a patch using `git diff` implicitly by changing files.
When you are done, submit your changes."""
        with open(baseline_file, 'w') as f:
            f.write(baseline_prompt)
        print(f"  (Baseline prompt saved to: {baseline_file})")

    # Print final cost summary
    print("\n" + "="*70)
    print("FINAL COST SUMMARY")
    print("="*70)
    final_summary = tracker.get_summary()
    print(final_summary)

    logger.info("Training session complete")
    logger.info(final_summary)

if __name__ == "__main__":
    main()
