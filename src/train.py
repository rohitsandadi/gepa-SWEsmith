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
from gepa.api import optimize
from gepa.utils.stop_condition import TimeoutStopCondition
from src.adapters.swe_adapter import SWEAdapter
from src.cost_tracker import reset_tracker
from src.experiment_logger import ExperimentLogger, set_logger

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=2, help="Number of GEPA generations")
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
    parser.add_argument("--stop-if-no-improvement", type=int, default=10, help="Stop after N iterations without improvement")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers (requires setup_parallel_workspaces.sh)")
    parser.add_argument("--max-metric-calls", type=int, default=600, help="Max rollouts for GEPA (controls budget)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--wandb-project", type=str, default="gepa-swesmith", help="Wandb project name")
    args = parser.parse_args()

    if args.smoke_test:
        args.generations = 1
        args.train_size = 2
        print("SMOKE TEST MODE")

    # Initialize experiment logger FIRST to get run_id
    # This creates a unique folder for this run: gepa_results/logs/run_YYYYMMDD_HHMMSS_<id>/
    exp_logger = ExperimentLogger(log_dir="gepa_results/logs")
    set_logger(exp_logger)
    run_dir = exp_logger.log_dir  # Use the run-specific directory

    # Setup Python logging to the run directory
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
    logger.info("="*70)
    logger.info(f"GEPA Training Session Started - Run: {exp_logger.run_id}")
    logger.info("="*70)

    if args.smoke_test:
        logger.info("Running in SMOKE TEST mode: 1 generation, 2 tasks")

    # Initialize cost tracker in the same run directory
    tracker = reset_tracker(log_dir=str(run_dir))
    print(f"\n💰 Cost tracking enabled")
    print(f"   Cost summary: {run_dir / 'cost_summary.txt'}\n")

    # 1. Load Data (train and validation for Pareto selection)
    if args.use_split:
        print("\n" + "="*70)
        print("Using pre-split datasets")
        print("="*70)
        train_data = load_split_data(split_name="train", limit=args.train_size)
        val_data = load_split_data(split_name="val", limit=args.val_size)
    else:
        print("\n" + "="*70)
        print("Loading data directly from HuggingFace (not using pre-split)")
        print("Recommendation: Run 'python split_dataset.py' and use --use-split flag")
        print("="*70)
        train_data = load_pygments_data_direct(limit=args.train_size)
        # Use subset for validation
        val_data = train_data[:min(args.val_size, len(train_data))]

    print(f"Loaded {len(train_data)} training, {len(val_data)} validation examples.")
    logger.info(f"Training data size: {len(train_data)}, Validation size: {len(val_data)}")

    # Determine reflection model (can be different from agent model)
    reflection_model = args.reflection_model or args.model

    # Save experiment config
    exp_logger.save_config({
        "model": args.model,
        "reflection_model": reflection_model,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "max_metric_calls": args.max_metric_calls,
        "workers": args.workers,
        "seed": args.seed,
        "timeout": args.timeout,
        "stop_if_no_improvement": args.stop_if_no_improvement,
        "use_split": args.use_split,
        "execution_mode": "docker",
        "wandb": args.wandb,
        "wandb_project": args.wandb_project if args.wandb else None,
    })

    # 2. Setup Adapter (Docker mode only)
    adapter = SWEAdapter(
        model_name=args.model, 
        n_workers=args.workers
    )
    logger.info(f"Model: {args.model}")
    logger.info(f"Execution: Docker containers via SWE-smith")

    # 3. Initial Candidate (The Baseline Prompt)
    # Start with empty skills - GEPA will learn and evolve them
    # The system_template in mini.yaml already has base instructions,
    # this just populates the {{ skills }} placeholder
    initial_skills = ""
    
    seed_candidate = {"skills": initial_skills}

    # 4. Setup stop conditions (NoImprovementStopper disabled - let max_metric_calls control budget)
    stop_callbacks = [
        TimeoutStopCondition(timeout_seconds=args.timeout),
    ]

    # 5. Run GEPA
    print("\n" + "="*70)
    print("Starting GEPA Optimization...")
    print("="*70)
    logger.info(f"Max metric calls: {args.max_metric_calls}")
    logger.info(f"Agent model: {args.model}")
    logger.info(f"Reflection model: {reflection_model}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Initial skills: {initial_skills[:100] if initial_skills else '(empty)'}...")

    print(f"\n📊 Watch progress:")
    print(f"   tail -f {log_file}")
    print(f"   tail -f {run_dir / 'cost_summary.txt'}")
    if args.wandb:
        print(f"   Wandb project: {args.wandb_project}")
        print(f"   Wandb run name: {exp_logger.run_id}")
    print()

    # Configure wandb to use same run ID as our logging
    wandb_kwargs = None
    if args.wandb:
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": exp_logger.run_id,
            "config": {
                "model": args.model,
                "reflection_model": reflection_model,
                "train_size": len(train_data),
                "val_size": len(val_data),
                "max_metric_calls": args.max_metric_calls,
                "workers": args.workers,
            }
        }

    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,  # Critical for Pareto selection
        adapter=adapter,
        # GEPA hyperparameters
        reflection_lm=reflection_model,  # Model for reflection (can differ from agent)
        max_metric_calls=args.max_metric_calls,  # Total rollouts (controls budget)
        reflection_minibatch_size=3,  # GEPA default (was 1)
        candidate_selection_strategy="pareto",  # Explicit Pareto selection
        skip_perfect_score=True,  # Skip reflection on perfect scores
        perfect_score=1.0,
        use_wandb=args.wandb,
        wandb_init_kwargs=wandb_kwargs,
        run_dir=str(run_dir),  # Use run-specific directory
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

    # Save best skills to run directory
    best_skills_file = run_dir / "best_skills.txt"
    with open(best_skills_file, 'w') as f:
        f.write(result.best_candidate.get("skills", ""))

    # Also save baseline (empty skills) for comparison
    baseline_skills = ""  # We start with empty skills
    baseline_file = run_dir / "baseline_skills.txt"
    with open(baseline_file, 'w') as f:
        f.write(baseline_skills)

    print(f"\n✓ Skills saved to run directory:")
    print(f"   Best: {best_skills_file}")
    print(f"   Baseline: {baseline_file}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on test set:")
    print(f"     python src/evaluate_prompts.py --split test --prompt {best_skills_file}")
    print(f"  2. Compare baseline vs optimized:")
    print(f"     python src/evaluate_prompts.py --baseline {baseline_file} --optimized {best_skills_file}")
    print(f"  3. Review full results in: {run_dir}/")

    # Print final cost summary
    print("\n" + "="*70)
    print("FINAL COST SUMMARY")
    print("="*70)
    tracker.print_summary()

    # Save experiment summary with before/after comparison
    best_skills = result.best_candidate.get("skills", "")
    exp_logger.save_summary(
        best_prompt=best_skills,  # experiment_logger still uses best_prompt param name
        extra_info={
            "model": args.model,
            "reflection_model": reflection_model,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "max_metric_calls": args.max_metric_calls,
            "workers": args.workers,
            "seed": args.seed,
        }
    )

    logger.info("Training session complete")

if __name__ == "__main__":
    main()
