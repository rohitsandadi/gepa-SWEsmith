"""
Fitness function for GEPA optimize_anything API.

This wraps the SWE harness logic into a simple fitness function that:
- Takes a candidate (prompt) and batch of tasks
- Returns (score, output, side_info) tuples for each task
"""

from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

from src.swe_harness import SWEHarness


def create_swe_fitness_fn(model_name: str = "gpt-4o", n_workers: int = 6):
    """Create a fitness function for SWE tasks with Docker containers.
    
    Args:
        model_name: LiteLLM model name
        n_workers: Number of parallel workers
        
    Returns:
        fitness_fn: Function that evaluates candidates on batches
    """
    
    # Create harness pool - one per worker
    harness_pool = [SWEHarness() for _ in range(n_workers)]
    harness_available = [True] * n_workers
    harness_lock = threading.Lock()
    
    def get_harness() -> Tuple[int, SWEHarness]:
        """Get an available harness from the pool."""
        with harness_lock:
            for i, available in enumerate(harness_available):
                if available:
                    harness_available[i] = False
                    return i, harness_pool[i]
        raise RuntimeError("No harness available - pool exhausted")
    
    def release_harness(idx: int):
        """Release a harness back to the pool."""
        with harness_lock:
            harness_available[idx] = True
    
    def process_single_task(
        task: Dict[str, Any],
        skills: str,
        task_idx: int,
        total_tasks: int
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """Process a single task and return (score, output, side_info)."""
        
        # Get a harness from the pool
        harness_idx, harness = get_harness()
        
        try:
            instance_id = task.get('instance_id', 'unknown')[:50]
            print(f"[Task {task_idx+1}/{total_tasks}] {instance_id}...", flush=True)
            
            # 1. Setup Docker Container
            harness.setup_task(task_instance=task)

            # 2. Run Agent
            problem = task["problem_statement"]
            patch, agent_trace, agent_metrics = harness.run_agent(
                problem, skills, model_name=model_name
            )
            

            # 3. Verify with tests
            has_patch = len(patch.strip()) > 0
            passed = False
            feedback_msg = ""
            test_output = ""

            if not has_patch:
                passed = False
                feedback_msg = "no_patch"
                test_output = "No patch to test."
            else:
                # FAIL_TO_PASS Tests - use repo profile's test command
                f2p_passed, f2p_output = harness.verify(f2p_only=True)
                test_output = f"=== FAIL_TO_PASS TESTS ===\n{f2p_output}"
                
                if not f2p_passed:
                    passed = False
                    feedback_msg = "f2p_failed"
                else:
                    # PASS_TO_PASS Tests (Regression Check) - run full test suite
                    pass_to_pass = task.get("PASS_TO_PASS", [])
                    
                    if pass_to_pass:
                        # Run full test command (includes both f2p and p2p)
                        p2p_passed, p2p_output = harness.verify(f2p_only=False)
                        test_output += f"\n\n=== FULL TEST SUITE ===\n{p2p_output}"
                        
                        if not p2p_passed:
                            passed = False
                            feedback_msg = "p2p_regression"
                        else:
                            passed = True
                            feedback_msg = "all_passed"
                    else:
                        passed = True
                        feedback_msg = "f2p_passed"

            score = 1.0 if passed else 0.0
            
            # Rollout output
            output = {
                "patch": patch,
                "success": passed,
                "steps": agent_metrics.get("steps", 0),
                "estimated_tokens": agent_metrics.get("estimated_tokens", 0)
            }
            
            # Side info for reflection
            side_info = {
                "Input": {
                    "Task ID": instance_id,
                    "Problem": problem[:200] + "..." if len(problem) > 200 else problem,
                },
                "Generated Outputs": {
                    "Patch": patch[:500] + "..." if len(patch) > 500 else patch,
                    "Agent Trace": agent_trace[:1000] + "..." if len(agent_trace) > 1000 else agent_trace,
                },
                "Feedback": {
                    "Status": feedback_msg,
                    "Test Output": test_output[:1000] + "..." if len(test_output) > 1000 else test_output,
                },
                "scores": {
                    "correctness": score,
                }
            }
            
            status = "✓ PASS" if passed else f"✗ FAIL ({feedback_msg})"
            print(f"  [{instance_id[:30]}] {status}", flush=True)
            
            # Cleanup
            harness.cleanup()
            
            return score, output, side_info
        
        finally:
            release_harness(harness_idx)
    
    def fitness_fn(
        candidate: Dict[str, str],
        batch: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
        """Evaluate candidate on batch of tasks.
        
        Args:
            candidate: Dict with 'skills' key
            batch: List of task instances
            
        Returns:
            List of (score, output, side_info) tuples
        """
        skills = candidate["skills"]
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for idx, task in enumerate(batch):
                future = executor.submit(
                    process_single_task,
                    task,
                    skills,
                    idx,
                    len(batch)
                )
                futures.append(future)
            
            # Collect results in order
            results = [future.result() for future in futures]
        
        return results
    
    return fitness_fn
