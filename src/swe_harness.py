import gc
import os
import shutil
import subprocess
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.local import LocalEnvironment
from minisweagent.config import get_config_path

@dataclass
class TaskResult:
    passed: bool
    trace: str
    output: str

class SWEHarness:
    def __init__(self, workspace_root: str = "/tmp/gepa_workenvs/pygments"):
        self.workspace_root = workspace_root
        
    def setup_task(self, base_commit: str, bug_patch: str = None):
        """Checkout the specific commit and apply the bug patch.

        SWE-smith workflow:
        1. Checkout base_commit (clean code)
        2. Apply bug_patch (introduces the synthetic bug)
        3. Now the code has a bug for the agent to fix
        """
        # Reset any changes aggressively
        subprocess.run(["git", "merge", "--abort"], cwd=self.workspace_root, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "restore", "."], cwd=self.workspace_root, check=False)
        subprocess.run(["git", "clean", "-fdx"], cwd=self.workspace_root, check=True)
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=self.workspace_root, check=True)
        # Checkout commit
        subprocess.run(["git", "checkout", "-f", base_commit], cwd=self.workspace_root, check=True)

        # Apply the bug patch (this introduces the bug the agent needs to fix)
        if bug_patch:
            result = subprocess.run(
                ["git", "apply", "--verbose"],
                input=bug_patch,
                cwd=self.workspace_root,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"  WARNING: Failed to apply bug patch: {result.stderr[:200]}")
            else:
                print(f"  Bug patch applied successfully")

    def run_agent(self, problem_statement: str, system_prompt: str, model_name: str = "gemini/gemma-3-4b-it") -> Tuple[str, str, dict]:
        """Run the agent and return (patch, conversation_trace, metrics).

        The system_prompt from GEPA is used as additional context/instructions
        that get prepended to the problem statement. This preserves mini-swe-agent's
        built-in templates which contain essential formatting instructions.

        Returns:
            patch: The git diff of changes made
            trace: Full conversation trace
            metrics: Dict with 'steps' (number of agent turns) and 'tokens' (estimated)
        """
        
        # Load the full mini-swe-agent config (includes proper system template with
        # file editing instructions, workflow guidance, etc.)
        config_path = get_config_path("mini")  # Gets path to mini.yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Extract only supported fields from agent config
        full_agent_config = config.get("agent", {})
        # DefaultAgent only accepts these template fields + limits
        supported_fields = [
            "system_template", "instance_template", "action_observation_template",
            "format_error_template", "timeout_template", "step_limit", "cost_limit"
        ]
        agent_config = {k: v for k, v in full_agent_config.items() if k in supported_fields}
        agent_config["step_limit"] = 30  # Reasonable limit for each task
        
        # Initialize Agent with the filtered config
        agent = DefaultAgent(
            model=LitellmModel(model_name=model_name),
            env=LocalEnvironment(cwd=self.workspace_root),
            **agent_config,
        )

        # Prepend our custom prompt to the problem statement
        # This way GEPA can optimize the instructions while keeping
        # mini-swe-agent's built-in action formatting
        enhanced_problem = f"""## Context
{system_prompt}

## Issue to Fix
{problem_statement}
"""

        try:
            # We wrap in try/except to ensure we capture trace even if it crashes
            result = agent.run(enhanced_problem)

            # Extract the full conversation trace from agent.messages
            # This contains the agent's reasoning, actions, and tool outputs
            trace = "\n\n".join([
                f"[{msg['role'].upper()}]\n{msg['content']}"
                for msg in agent.messages
            ])

            # Calculate metrics
            num_steps = len([m for m in agent.messages if m.get('role') == 'assistant'])
            
            # Use LiteLLM's token counter for accurate count
            try:
                import litellm
                # token_counter expects messages with 'role' and 'content' keys
                token_count = litellm.token_counter(model=model_name, messages=agent.messages)
            except Exception:
                # Fallback to estimate if tokenizer fails
                total_chars = sum(len(m.get('content', '')) for m in agent.messages)
                token_count = total_chars // 4

            metrics = {
                "steps": num_steps,
                "estimated_tokens": token_count,
                "num_messages": len(agent.messages)
            }

            # Generate patch of changes
            diff_proc = subprocess.run(
                ["git", "diff"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True
            )
            patch = diff_proc.stdout

            # Explicit cleanup to prevent memory leaks
            del agent
            gc.collect()

            return patch, trace, metrics

        except Exception as e:
            gc.collect()  # Clean up even on error
            return "", f"Agent crashed: {str(e)}", {"steps": 0, "estimated_tokens": 0, "num_messages": 0}

    def verify(self, test_cmd: str = "pytest") -> Tuple[bool, str]:
        """Run verification tests and return (passed, output).

        Returns both the pass/fail status and the full test output,
        which has info like:
        - Test failure messages
        - Stack traces
        - Compilation errors
        - Expected vs actual values
        """
        try:
            proc = subprocess.run(
                test_cmd,
                cwd=self.workspace_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120  # Allow 2 minutes for larger test suites
            )

            # Combine stdout and stderr for full diagnostic output
            test_output = f"=== TEST COMMAND: {test_cmd} ===\n"
            test_output += f"=== RETURN CODE: {proc.returncode} ===\n\n"

            if proc.stdout:
                test_output += "=== STDOUT ===\n" + proc.stdout + "\n"
            if proc.stderr:
                test_output += "=== STDERR ===\n" + proc.stderr + "\n"

            passed = proc.returncode == 0
            return passed, test_output

        except subprocess.TimeoutExpired:
            return False, f"TEST TIMEOUT: Command '{test_cmd}' exceeded 60 second limit"

    def cleanup(self):
        """Cleanup after run."""
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=self.workspace_root, check=False)
        subprocess.run(["git", "checkout", "-f", "master"], cwd=self.workspace_root, check=False)
