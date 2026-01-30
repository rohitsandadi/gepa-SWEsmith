"""
Evaluate and compare prompts on held-out test set.
Supports: baseline, GEPA-optimized, and pure mini-swe-agent (no context).
Integrates with ExperimentLogger for structured logging.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from src.adapters.swe_adapter import SWEAdapter
from src.experiment_logger import ExperimentLogger, set_logger, get_logger

def load_split(split_name="test"):
    """Load a pre-split dataset."""
    data_file = Path("data") / f"pygments_{split_name}.json"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {data_file}\n"
            f"Run: python split_dataset.py first"
        )

    with open(data_file) as f:
        return json.load(f)

def verify_test_set_integrity():
    """Verify test set hasn't been contaminated by training."""
    split_ids_file = Path("data/split_instance_ids.json")
    if not split_ids_file.exists():
        print("WARNING: Cannot verify test set integrity - split_instance_ids.json not found")
        return True

    with open(split_ids_file) as f:
        splits = json.load(f)

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    # Check for overlaps
    if train_ids & test_ids:
        print("ERROR: Test set overlaps with training set!")
        return False
    if val_ids & test_ids:
        print("ERROR: Test set overlaps with validation set!")
        return False

    print(f"✓ Test set verified: {len(test_ids)} tasks, no overlap with train/val")
    return True

def evaluate_prompt(
    prompt_text,
    prompt_name,
    data,
    model_name="gpt-5.2",
    adapter=None
):
    """
    Evaluate a prompt on provided data.
    Returns (pass_rate, results).
    """
    if adapter is None:
        adapter = SWEAdapter(
            workspace_root="/tmp/gepa_workenvs/pygments",
            model_name=model_name
        )

    # Evaluate
    candidate = {"skills": prompt_text}
    eval_batch = adapter.evaluate(
        batch=data,
        candidate=candidate,
        capture_traces=False
    )

    # Calculate metrics
    scores = eval_batch.scores
    outputs = eval_batch.outputs
    pass_count = sum(1 for s in scores if s == 1.0)
    total = len(scores)
    pass_rate = (pass_count / total) * 100 if total > 0 else 0

    # Build detailed results
    results = []
    for i, (task, score, output) in enumerate(zip(data, scores, outputs)):
        results.append({
            "instance_id": task["instance_id"],
            "passed": score == 1.0,
            "score": score,
            "steps": output.get("steps", 0),
            "tokens": output.get("estimated_tokens", 0),
            "has_patch": len(output.get("patch", "")) > 0
        })

    # Log to experiment logger if available
    logger = get_logger()
    if logger:
        logger.log_eval_batch(
            prompt=prompt_text,
            outputs=outputs,
            scores=scores,
            task_ids=[task["instance_id"] for task in data],
            is_baseline=(prompt_name == "mini_swe_agent_default")
        )

    return pass_rate, results, outputs, scores

def run_evaluation(args):
    """Main evaluation function."""

    # Initialize experiment logger
    run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = ExperimentLogger(log_dir="gepa_results/eval_logs", run_name=run_name)
    set_logger(logger)

    # Save config
    config = {
        "type": "evaluation",
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "model": args.model,
        "workers": args.workers,
        "training_run_id": args.run_id,
        "prompts_to_evaluate": []
    }

    # Verify test set if needed
    if args.split == "test":
        if not verify_test_set_integrity():
            raise ValueError("Test set integrity check failed!")

    # Load data
    data = load_split(args.split)
    total_available = len(data)

    # Apply offset and limit
    data = data[args.offset:]
    if args.limit:
        data = data[:args.limit]

    print(f"\n{'='*70}")
    print(f"EVALUATION RUN: {run_name}")
    print(f"Split: {args.split} | Tasks: {len(data)} (offset={args.offset}, available={total_available})")
    print(f"Model: {args.model} | Workers: {args.workers}")
    if args.run_id:
        print(f"Training run: {args.run_id}")
    print(f"{'='*70}")

    # Create shared adapter
    adapter = SWEAdapter(
        workspace_root="/tmp/gepa_workenvs/pygments",
        model_name=args.model,
        n_workers=args.workers
    )

    results_summary = {}
    all_task_results = {}

    # Define prompts to evaluate
    prompts_to_eval = []

    # Mini-SWE-agent default (empty context - just uses mini.yaml system template)
    if args.mini_swe_agent or args.all:
        prompts_to_eval.append({
            "name": "mini_swe_agent_default",
            "prompt": "",  # Empty = pure mini-swe-agent
            "description": "Pure mini-swe-agent (no additional context)"
        })
        config["prompts_to_evaluate"].append("mini_swe_agent_default")

    # Baseline prompt
    if args.baseline:
        with open(args.baseline) as f:
            baseline_prompt = f.read().strip()
        prompts_to_eval.append({
            "name": "baseline",
            "prompt": baseline_prompt,
            "description": f"Baseline from {args.baseline}"
        })
        config["prompts_to_evaluate"].append("baseline")

    # Optimized prompt
    if args.optimized:
        with open(args.optimized) as f:
            optimized_prompt = f.read().strip()
        prompts_to_eval.append({
            "name": "gepa_optimized",
            "prompt": optimized_prompt,
            "description": f"GEPA-optimized from {args.optimized}"
        })
        config["prompts_to_evaluate"].append("gepa_optimized")

    logger.save_config(config)

    # Run evaluations
    for i, prompt_info in enumerate(prompts_to_eval):
        name = prompt_info["name"]
        prompt = prompt_info["prompt"]
        desc = prompt_info["description"]

        print(f"\n[{i+1}/{len(prompts_to_eval)}] Evaluating: {name}")
        print(f"    {desc}")
        print("-" * 70)

        pass_rate, results, outputs, scores = evaluate_prompt(
            prompt, name, data, args.model, adapter
        )

        results_summary[name] = {
            "pass_rate": pass_rate,
            "passed": sum(1 for r in results if r["passed"]),
            "total": len(results),
            "avg_steps": sum(r["steps"] for r in results) / len(results) if results else 0,
            "avg_tokens": sum(r["tokens"] for r in results) / len(results) if results else 0,
        }
        all_task_results[name] = results

        print(f"\n✓ {name}: {results_summary[name]['passed']}/{results_summary[name]['total']} passed ({pass_rate:.1f}%)")

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    for name, summary in results_summary.items():
        print(f"\n{name}:")
        print(f"  Pass rate: {summary['pass_rate']:.1f}% ({summary['passed']}/{summary['total']})")
        print(f"  Avg steps: {summary['avg_steps']:.1f}")
        print(f"  Avg tokens: {summary['avg_tokens']:.0f}")

    # Calculate improvements
    if len(results_summary) >= 2:
        names = list(results_summary.keys())
        base_name = names[0]
        for other_name in names[1:]:
            base_rate = results_summary[base_name]["pass_rate"]
            other_rate = results_summary[other_name]["pass_rate"]
            diff = other_rate - base_rate
            print(f"\n{other_name} vs {base_name}: {diff:+.1f} pp")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = logger.log_dir / f"evaluation_results_{timestamp}.json"

    final_results = {
        "run_id": run_name,
        "training_run_id": args.run_id,
        "timestamp": timestamp,
        "split": args.split,
        "model": args.model,
        "offset": args.offset,
        "limit": args.limit,
        "total_tasks": len(data),
        "task_ids": [t["instance_id"] for t in data],
        "summary": results_summary,
        "per_task_results": all_task_results
    }

    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n📊 Detailed results saved to: {results_file}")

    # Save experiment summary
    logger.save_summary(extra_info={"evaluation_summary": results_summary})

    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate prompts on held-out test set")

    # Prompt sources
    parser.add_argument("--baseline", type=str, help="Path to baseline prompt file")
    parser.add_argument("--optimized", type=str, help="Path to GEPA-optimized prompt file")
    parser.add_argument("--mini-swe-agent", action="store_true",
                        help="Include pure mini-swe-agent (no additional context)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all: mini-swe-agent default + baseline + optimized")

    # Data selection
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N tasks (for resuming)")

    # Model config
    parser.add_argument("--model", type=str, default="gpt-5.2",
                        help="Model to use for evaluation")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of parallel workers (default: 3)")

    # Metadata
    parser.add_argument("--run-id", type=str,
                        help="ID of the training run being evaluated")

    args = parser.parse_args()

    # Validate arguments
    if not (args.baseline or args.optimized or args.mini_swe_agent or args.all):
        parser.error("At least one of --baseline, --optimized, --mini-swe-agent, or --all is required")

    # If --all, set the individual flags
    if args.all:
        args.mini_swe_agent = True
        if not args.baseline:
            args.baseline = "gepa_results/baseline_prompt.txt"
        if not args.optimized:
            args.optimized = "gepa_results/logs/latest_backup_349rollouts/prompts/prompt_5b1dcd6d.txt"

    run_evaluation(args)

if __name__ == "__main__":
    main()
