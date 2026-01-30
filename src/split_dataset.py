"""
Pre-split the Pygments dataset into train/val/test sets.
This ensures reproducible experiments and proper held-out evaluation.
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

def split_pygments_dataset(
    total_limit=1200,  # Larger dataset for better evaluation
    train_ratio=0.65,  # 65% train (~780 tasks)
    val_ratio=0.05,    # 5% validation (~60 tasks for GEPA development)
    test_ratio=0.30,   # 30% test (~360 tasks for final evaluation)
    seed=42,
    # Explicit sizes (override ratios if provided)
    explicit_train_size=None,
    explicit_val_size=None,
    explicit_test_size=None,
):
    """
    Load Pygments tasks and split into train/val/test sets.
    Saves to data/ directory for reproducibility.

    If explicit sizes are provided, they override the ratios.
    """

    # If explicit sizes provided, calculate total needed
    if explicit_train_size and explicit_val_size and explicit_test_size:
        total_limit = explicit_train_size + explicit_val_size + explicit_test_size
        print(f"Using explicit sizes: {explicit_train_size}/{explicit_val_size}/{explicit_test_size} = {total_limit} total")

    print(f"Loading Pygments tasks from SWE-smith...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")

    # Filter to Pygments tasks
    target_repo = "swesmith/pygments__pygments"
    pygments_tasks = []

    for item in ds:
        repo_name = item.get('repo', '')
        if target_repo in repo_name:
            pygments_tasks.append(item)
            if len(pygments_tasks) >= total_limit:
                break

    print(f"Collected {len(pygments_tasks)} Pygments tasks")

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(pygments_tasks)

    # Calculate split sizes (explicit sizes override ratios)
    n = len(pygments_tasks)
    if explicit_train_size and explicit_val_size and explicit_test_size:
        train_size = explicit_train_size
        val_size = explicit_val_size
        test_size = explicit_test_size
    else:
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        test_size = n - train_size - val_size

    # Split the data
    train_data = pygments_tasks[:train_size]
    val_data = pygments_tasks[train_size:train_size + val_size]
    test_data = pygments_tasks[train_size + val_size:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} tasks ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_data)} tasks ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_data)} tasks ({test_ratio*100:.0f}%)")

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        output_file = data_dir / f"pygments_{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"  Saved {split_name} to {output_file}")

    # Save metadata
    metadata = {
        'total_tasks': len(pygments_tasks),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'seed': seed,
        'target_repo': target_repo,
        'explicit_sizes_used': explicit_train_size is not None,
    }

    metadata_file = data_dir / "split_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_file}")

    # Save instance IDs separately for easy verification (in order)
    instance_ids = {
        'train': [t.get('instance_id', f'unknown_{i}') for i, t in enumerate(train_data)],
        'val': [t.get('instance_id', f'unknown_{i}') for i, t in enumerate(val_data)],
        'test': [t.get('instance_id', f'unknown_{i}') for i, t in enumerate(test_data)],
    }
    ids_file = data_dir / "split_instance_ids.json"
    with open(ids_file, 'w') as f:
        json.dump(instance_ids, f, indent=2)
    print(f"  Saved instance IDs to {ids_file}")

    print(f"\n✓ Dataset split complete!")
    print(f"\nUsage:")
    print(f"  - Use train split for GEPA optimization")
    print(f"  - Use val split for selecting best prompt during optimization")
    print(f"  - Use test split ONLY for final evaluation (never during training!)")

    return train_data, val_data, test_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1200,
                        help="Total number of tasks to use (default: 1200 = ~780 train, ~60 val, ~360 test)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.65,
                        help="Fraction for training (default: 0.65)")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction for validation (default: 0.05)")
    parser.add_argument("--test-ratio", type=float, default=0.30,
                        help="Fraction for test (default: 0.30)")
    # Explicit sizes (override ratios)
    parser.add_argument("--train-size", type=int, default=None,
                        help="Explicit train size (overrides --train-ratio)")
    parser.add_argument("--val-size", type=int, default=None,
                        help="Explicit val size (overrides --val-ratio)")
    parser.add_argument("--test-size", type=int, default=None,
                        help="Explicit test size (overrides --test-ratio)")
    args = parser.parse_args()

    split_pygments_dataset(
        total_limit=args.total,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        explicit_train_size=args.train_size,
        explicit_val_size=args.val_size,
        explicit_test_size=args.test_size,
    )
