from typing import Any, Mapping, Sequence, List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import threading
import os

from gepa.core.adapter import EvaluationBatch
from src.swe_harness import SWEHarness
from src.experiment_logger import get_logger

# Define types
DataInst = Dict[str, Any]  # SWESmith task instance
Trajectory = Dict[str, Any] # Agent trace and logs
RolloutOutput = Dict[str, Any] # Patch and status

class SWEAdapter:
    def __init__(
        self, 
        model_name: str = "gpt-4o",
        n_workers: int = 6
    ):
        self.model_name = model_name
        self.n_workers = n_workers
        self.propose_new_texts = None  # Default GEPA proposer
        # Cost tracking is automatic via LiteLLM callbacks in cost_tracker.py
        self.task_count = 0
        self._task_count_lock = threading.Lock()
        self.iteration_count = 0  # Track GEPA iterations for logging
        
        # Create harness pool - one per worker
        # Each harness will create Docker containers on demand
        self.harness_pool = [SWEHarness() for _ in range(n_workers)]
        
        # Track which harnesses are available
        self._harness_available = [True] * n_workers
        self._harness_lock = threading.Lock()
        
        print(f"Initialized SWEAdapter with {n_workers} workers (Docker mode)")

    def _get_harness(self) -> tuple[int, SWEHarness]:
        """Get an available harness from the pool."""
        with self._harness_lock:
            for i, available in enumerate(self._harness_available):
                if available:
                    self._harness_available[i] = False
                    return i, self.harness_pool[i]
        # Should not reach here if pool size >= n_workers
        raise RuntimeError("No harness available - pool exhausted")

    def _release_harness(self, idx: int):
        """Release a harness back to the pool."""
        with self._harness_lock:
            self._harness_available[idx] = True

    def _process_single_task(
        self,
        task: DataInst,
        skills: str,
        capture_traces: bool,
        task_idx: int,
        total_tasks: int
    ) -> tuple[RolloutOutput, float, Trajectory | None]:
        """Process a single task. Thread-safe."""
        
        # Get a harness from the pool
        harness_idx, harness = self._get_harness()
        
        try:
            with self._task_count_lock:
                self.task_count += 1
                current_count = self.task_count
            
            instance_id = task.get('instance_id', 'unknown')[:50]
            print(f"[Task {task_idx+1}/{total_tasks}] {instance_id}...", flush=True)
            
            # 1. Setup Docker Container
            # SWE-smith creates a container with the bug already applied
            harness.setup_task(task_instance=task)

            # 2. Run Agent
            problem = task["problem_statement"]
            patch, agent_reasoning_trace, agent_metrics = harness.run_agent(
                problem, skills, model_name=self.model_name
            )

            # 3. Verify with tests
            has_patch = len(patch.strip()) > 0
            passed = False
            feedback_msg = ""
            test_verification_output = ""

            if not has_patch:
                passed = False
                feedback_msg = "Agent did not produce any valid patch (git diff was empty)."
                test_verification_output = "No patch to test."
            else:
                # Use SWE-smith's run_patch_in_container for proper verification
                f2p_passed, f2p_output = harness.verify_with_patch(patch, f2p_only=True)
                test_verification_output = f"=== FAIL_TO_PASS TESTS ===\n{f2p_output}"
                
                if not f2p_passed:
                    passed = False
                    feedback_msg = "FAIL_TO_PASS tests failed - the fix does not solve the issue."
                else:
                    # PASS_TO_PASS Tests (Regression Check)
                    pass_to_pass = task.get("PASS_TO_PASS", [])
                    
                    if pass_to_pass:
                        # Run full test (includes both f2p and p2p)
                        p2p_passed, p2p_output = harness.verify_with_patch(patch, f2p_only=False)
                        test_verification_output += f"\n\n=== FULL TEST SUITE ===\n{p2p_output}"
                        
                        if not p2p_passed:
                            passed = False
                            feedback_msg = "PASS_TO_PASS regression failed - the fix breaks existing tests."
                        else:
                            passed = True
                            feedback_msg = "All tests passed (FAIL_TO_PASS fixed + PASS_TO_PASS still passing)."
                    else:
                        passed = True
                        feedback_msg = "FAIL_TO_PASS tests passed. No PASS_TO_PASS tests to check."

            score = 1.0 if passed else 0.0
            output = {
                "patch": patch,
                "success": passed,
                "steps": agent_metrics.get("steps", 0),
                "estimated_tokens": agent_metrics.get("estimated_tokens", 0)
            }
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  [{instance_id[:30]}] {status}", flush=True)

            trajectory = None
            if capture_traces:
                trajectory = {
                    "agent_reasoning_trace": agent_reasoning_trace,
                    "test_verification_output": test_verification_output,
                    "patch": patch,
                    "feedback": feedback_msg,
                    "instance_id": task["instance_id"],
                    "problem_statement": task["problem_statement"],
                    "metrics": agent_metrics  # Track steps and tokens
                }

            # Cleanup
            harness.cleanup()
            
            return output, score, trajectory
            
        finally:
            self._release_harness(harness_idx)

    def evaluate(
        self,
        batch: List[DataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        
        outputs: List[RolloutOutput] = []
        scores: List[float] = []
        trajectories: List[Trajectory] | None = [] if capture_traces else None
        
        skills = candidate.get("skills", "")

        print(f"\n{'='*70}")
        print(f"Evaluating batch of {len(batch)} tasks with {self.n_workers} workers...")
        print(f"{'='*70}\n")

        # Use ThreadPoolExecutor for parallel execution
        results = [None] * len(batch)  # Preserve order
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    self._process_single_task,
                    task, skills, capture_traces, i, len(batch)
                ): i
                for i, task in enumerate(batch)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    output, score, trajectory = future.result()
                    results[idx] = (output, score, trajectory)
                except Exception as e:
                    print(f"  Task {idx} failed with error: {e}")
                    results[idx] = ({"patch": "", "success": False}, 0.0, None)
        
        # Unpack results in order
        for output, score, trajectory in results:
            outputs.append(output)
            scores.append(score)
            if capture_traces:
                trajectories.append(trajectory)  # Append even if None to keep indices aligned

        # Print batch summary with metrics
        pass_count = sum(1 for s in scores if s == 1.0)
        avg_steps = sum(o.get("steps", 0) for o in outputs) / len(outputs) if outputs else 0
        avg_tokens = sum(o.get("estimated_tokens", 0) for o in outputs) / len(outputs) if outputs else 0
        print(f"\nBatch complete: {pass_count}/{len(batch)} passed ({pass_count/len(batch)*100:.1f}%)")
        print(f"Avg steps: {avg_steps:.1f}, Avg tokens: {avg_tokens:.0f}")
        print(f"{'='*70}\n")

        # Log evaluation batch metrics
        logger = get_logger()
        if logger:
            task_ids = [task.get("instance_id", f"task_{i}") for i, task in enumerate(batch)]
            is_baseline = self.iteration_count == 0
            logger.log_eval_batch(
                prompt=skills,
                outputs=outputs,
                scores=scores,
                task_ids=task_ids,
                is_baseline=is_baseline
            )
            self.iteration_count += 1

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Prepare reflection data for GEPA.

        This combines TWO sources of feedback (as per GEPA paper Section 3.2):
        1. Agent reasoning traces (LLM's chain of thought, tool calls, decisions)
        2. Environment/evaluation feedback (test output, errors, stack traces)

        GEPA uses this rich textual feedback to reflect on failures and propose
        targeted prompt improvements - this is what gives it an advantage over RL.
        """
        updates = []

        if not eval_batch.trajectories:
            return {"skills": []}

        for i, traj in enumerate(eval_batch.trajectories):
            # Skip None trajectories (from tasks that threw exceptions)
            if traj is None:
                continue

            # Combine agent reasoning + environment feedback
            # Truncate to avoid token limits while preserving diagnostic info
            agent_trace_truncated = traj["agent_reasoning_trace"][:10000]
            test_output_truncated = traj["test_verification_output"][:5000]

            record = {
                "Inputs": {
                    "Task ID": traj["instance_id"],
                    "Problem Statement": traj["problem_statement"][:500] + "..."
                },
                "Generated Outputs": {
                    "Agent Reasoning & Actions": agent_trace_truncated,
                    "Patch Generated": traj["patch"][:2000] if traj["patch"] else "No patch generated"
                },
                "Environment Feedback": {
                    "Test Verification Output": test_output_truncated,
                    "Status": traj['feedback'],
                    "Score": eval_batch.scores[i]
                }
            }
            updates.append(record)

        # Log proposer input for analysis
        logger = get_logger()
        if logger and updates:
            current_skills = candidate.get("skills", "")
            logger.log_proposer_input(
                iteration=self.iteration_count,
                current_prompt=current_skills,
                reflection_records=updates
            )

        return {"skills": updates}
