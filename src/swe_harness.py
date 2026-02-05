import gc
import json
import os
import platform
import subprocess
import threading
import yaml
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Suppress verbose LiteLLM logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import litellm
litellm.suppress_debug_info = True

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig

from swesmith.profiles import registry
from swesmith.harness.utils import run_patch_in_container
from swebench.harness.constants import KEY_INSTANCE_ID, LOG_TEST_OUTPUT, RUN_EVALUATION_LOG_DIR
import docker


# =============================================================================
# Docker Image Cache for Claude Code (Thread-Safe)
# =============================================================================
# Maps base_image_id (short) -> claude_code_image_name
# This is a module-level cache shared across all SWEHarness instances
_CLAUDE_CODE_IMAGE_CACHE: Dict[str, str] = {}
_CLAUDE_CODE_IMAGE_LOCK = threading.Lock()  # Protects cache access and image building


class ExistingContainerEnvironment(DockerEnvironment):
    """DockerEnvironment subclass that uses an existing container instead of creating one."""
    
    def __init__(self, container_id: str, cwd: str = "/testbed", timeout: int = 120):
        # Don't call super().__init__() - it would try to create a new container
        # Just set up the minimal config needed
        self.logger = None
        self.container_id = container_id
        self.config = DockerEnvironmentConfig(
            image="unused",  # Not used since we already have a container
            cwd=cwd,
            timeout=timeout,
            env={
                'PAGER': 'cat',
                'MANPAGER': 'cat', 
                'LESS': '-R',
                'PIP_PROGRESS_BAR': 'off',
                'TQDM_DISABLE': '1',
            },
        )
    
    def cleanup(self):
        """No cleanup - container is managed by SWEHarness."""
        pass

@dataclass
class TaskResult:
    passed: bool
    trace: str
    output: str


class SWEHarness:
    """Harness for running SWE tasks in Docker containers.
    
    Supports multiple agent types:
    - minisweagent: Default mini-swe-agent (LiteLLM-based)
    - claude-code: Anthropic's Claude Code CLI
    """
    
    # Supported agent types
    AGENT_TYPES = ["minisweagent", "claude-code"]
    
    def __init__(self, agent_type: str = "minisweagent"):
        """Initialize harness with SWE-smith Docker containers.
        
        Args:
            agent_type: Type of agent to use ("minisweagent" or "claude-code")
        """
        if agent_type not in self.AGENT_TYPES:
            raise ValueError(f"Unknown agent_type: {agent_type}. Must be one of {self.AGENT_TYPES}")
        
        self.agent_type = agent_type
        self.container = None
        self.repo_profile = None
        self.current_task = None
        
        # Set DOCKER_HOST environment variable for rootless Docker
        # This ensures SWE-smith's internal docker.from_env() calls work correctly
        if not os.getenv('DOCKER_HOST'):
            # Try to detect rootless Docker
            uid = os.getuid()
            xdg_runtime = os.getenv('XDG_RUNTIME_DIR')
            
            if xdg_runtime:
                rootless_socket = f"unix://{xdg_runtime}/docker.sock"
            else:
                rootless_socket = f"unix:///run/user/{uid}/docker.sock"
            
            # Check if rootless socket exists
            import pathlib
            socket_path = rootless_socket.replace('unix://', '')
            if pathlib.Path(socket_path).exists():
                os.environ['DOCKER_HOST'] = rootless_socket
                print(f"  Using rootless Docker: {rootless_socket}")
        
        # Try to connect to Docker
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Docker: {e}\n\n"
                f"Please ensure Docker is running:\n"
                f"  Rootless: systemctl --user start docker\n"
                f"  Standard: sudo systemctl start docker\n\n"
                f"If using rootless Docker, ensure DOCKER_HOST is set:\n"
                f"  export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock"
            ) from e
        
    def _get_claude_code_image(self, base_image_id: str) -> Optional[str]:
        """Check if a Claude Code-enabled image exists for this base image.
        
        Thread-safe: uses lock for cache access.
        
        Args:
            base_image_id: Short ID of the base image (first 12 chars)
            
        Returns:
            Cached image name if exists, None otherwise
        """
        global _CLAUDE_CODE_IMAGE_CACHE, _CLAUDE_CODE_IMAGE_LOCK
        
        with _CLAUDE_CODE_IMAGE_LOCK:
            # Check in-memory cache first
            if base_image_id in _CLAUDE_CODE_IMAGE_CACHE:
                return _CLAUDE_CODE_IMAGE_CACHE[base_image_id]
            
            # Check if image exists in Docker
            cached_name = f"gepa-claude-code-{base_image_id}"
            try:
                self.docker_client.images.get(cached_name)
                _CLAUDE_CODE_IMAGE_CACHE[base_image_id] = cached_name
                return cached_name
            except docker.errors.ImageNotFound:
                return None
    
    def _build_claude_code_image(self, base_image_id: str) -> str:
        """Build a Claude Code-enabled Docker image from a base image.
        
        Thread-safe: uses lock to prevent multiple threads from building
        the same image simultaneously.
        
        Creates a new image with Claude Code pre-installed, tagged as
        gepa-claude-code-{base_image_id}.
        
        Args:
            base_image_id: Short ID of the base image
            
        Returns:
            Name of the cached image
        """
        global _CLAUDE_CODE_IMAGE_CACHE, _CLAUDE_CODE_IMAGE_LOCK
        
        cached_name = f"gepa-claude-code-{base_image_id}"
        
        with _CLAUDE_CODE_IMAGE_LOCK:
            # Check if another thread already built it
            if base_image_id in _CLAUDE_CODE_IMAGE_CACHE:
                print(f"  Claude Code image already exists: {cached_name}")
                return _CLAUDE_CODE_IMAGE_CACHE[base_image_id]
            
            # Also check Docker in case it exists but not in memory cache
            try:
                self.docker_client.images.get(cached_name)
                _CLAUDE_CODE_IMAGE_CACHE[base_image_id] = cached_name
                print(f"  Claude Code image already exists: {cached_name}")
                return cached_name
            except docker.errors.ImageNotFound:
                pass
            
            print(f"  Building Claude Code image: {cached_name}...")
            
            # Create temporary container from base image
            temp_container = self.docker_client.containers.run(
                base_image_id,
                command="sleep 3600",
                detach=True,
                remove=False,
            )
            
            try:
                # Install Claude Code via npm
                # Note: Most SWE-smith images have Node.js installed
                print("    Installing Claude Code (npm install -g @anthropic-ai/claude-code)...")
                install_result = temp_container.exec_run(
                    "npm install -g @anthropic-ai/claude-code",
                    environment={"HOME": "/root"},
                    workdir="/root",
                )
                
                if install_result.exit_code != 0:
                    output = install_result.output.decode() if install_result.output else ""
                    raise RuntimeError(f"Failed to install Claude Code:\n{output}")
                
                # Verify installation
                verify_result = temp_container.exec_run("claude --version")
                if verify_result.exit_code != 0:
                    raise RuntimeError("Claude Code installed but 'claude --version' failed")
                
                version = verify_result.output.decode().strip()
                print(f"    Installed Claude Code version: {version}")
                
                # Commit container as new image
                temp_container.commit(repository=cached_name)
                print(f"    Cached image: {cached_name}")
                
            finally:
                temp_container.stop()
                temp_container.remove()
            
            _CLAUDE_CODE_IMAGE_CACHE[base_image_id] = cached_name
            return cached_name
    
    def _ensure_claude_code_installed(self):
        """Ensure Claude Code is installed in the current container.
        
        Thread-safe: uses lock to prevent multiple threads from installing
        simultaneously. If not installed, installs it and caches the image.
        """
        global _CLAUDE_CODE_IMAGE_CACHE, _CLAUDE_CODE_IMAGE_LOCK
        
        # Quick check without lock - if already installed, skip
        result = self.container.exec_run("which claude")
        if result.exit_code == 0:
            print("    Claude Code already installed")
            return
        
        # Not installed - acquire lock to prevent race condition
        # where multiple threads try to install simultaneously
        base_image_id = self.container.image.id[:12]
        
        with _CLAUDE_CODE_IMAGE_LOCK:
            # Double-check after acquiring lock (another thread may have installed)
            result = self.container.exec_run("which claude")
            if result.exit_code == 0:
                print("    Claude Code already installed (by another thread)")
                return
            
            print(f"    Claude Code not in container, installing...")
            
            # Install
            install_result = self.container.exec_run(
                "npm install -g @anthropic-ai/claude-code",
                environment={"HOME": "/root"},
                workdir="/root",
            )
            
            if install_result.exit_code != 0:
                output = install_result.output.decode() if install_result.output else ""
                raise RuntimeError(f"Failed to install Claude Code:\n{output}")
            
            # Verify
            verify_result = self.container.exec_run("claude --version")
            if verify_result.exit_code == 0:
                version = verify_result.output.decode().strip()
                print(f"    Installed Claude Code: {version}")
            
            # Cache image for future use
            # Note: This caches the current state INCLUDING the repo setup,
            # which won't be reusable for other tasks. But it's better than nothing.
            cached_name = f"gepa-claude-code-{base_image_id}"
            try:
                self.container.commit(repository=cached_name)
                _CLAUDE_CODE_IMAGE_CACHE[base_image_id] = cached_name
                print(f"    Cached image: {cached_name}")
            except Exception as e:
                print(f"    Warning: Failed to cache image: {e}")

    def setup_task(self, task_instance: Dict[str, Any]):
        """Setup task environment using SWE-smith Docker container.
        
        Args:
            task_instance: Full SWE-smith task instance
        """
        # Cleanup previous container if any
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except:
                pass
        
        # Get container from SWE-smith
        # The container comes with:
        # - Repository cloned at /testbed
        # - Instance branch checked out at HEAD (tests removed - agent can't see them)
        # - All dependencies installed
        # 
        # NOTE: SWE-smith branch structure:
        # - HEAD: Bug commit WITHOUT test files (for agent work)
        # - HEAD~1: Bug commit WITH test files (for evaluation)
        # The agent works at HEAD. Verification uses run_patch_in_container which
        # handles the proper checkout to HEAD~1 for testing.
        self.repo_profile = registry.get_from_inst(task_instance)
        self.current_task = task_instance
        self.container = self.repo_profile.get_container(task_instance)
        print(f"  Docker container created: {self.container.id[:12]}")
        
        # For Claude Code agent, ensure it's installed
        if self.agent_type == "claude-code":
            self._ensure_claude_code_installed()
        
    def run_agent(self, problem_statement: str, skills: str, model_name: str = "gemini/gemini-2.0-flash-exp", config_path: str = None) -> Tuple[str, str, dict]:
        """Run the agent in Docker container and return (patch, conversation_trace, metrics).

        The skills from GEPA are injected into the system template's {{ skills }} placeholder.
        This allows GEPA to evolve the agent's learned skills over time.

        Args:
            problem_statement: The problem to solve
            skills: Skills string to inject (ignored if config doesn't have {{ skills }} placeholder)
            model_name: LiteLLM model name
            config_path: Optional custom config path. Defaults to mini.yaml

        Returns:
            patch: The git diff of changes made
            trace: Full conversation trace
            metrics: Dict with 'steps' (number of agent turns) and 'tokens' (estimated)
        """
        
        # Load our custom mini-swe-agent config from the project directory
        # This config has {{ skills }} placeholder that GEPA will optimize
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "mini_swe_agent_config", "mini.yaml")
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
        agent_config["step_limit"] = 50  # Max steps per task
        
        # Get model kwargs from config and add OpenAI regional endpoint if needed
        model_config = config.get("model", {})
        model_kwargs = model_config.get("model_kwargs", {}).copy()
        
        # Add OpenAI regional endpoint (us.api.openai.com) if using OpenAI models
        if "openai" in model_name.lower() or model_name.startswith("gpt-"):
            if "api_base" not in model_kwargs:
                model_kwargs["api_base"] = "https://us.api.openai.com/v1"
        
        # Initialize Agent with Docker container environment
        # Use ExistingContainerEnvironment to execute commands inside the container
        agent = DefaultAgent(
            model=LitellmModel(model_name=model_name, model_kwargs=model_kwargs),
            env=ExistingContainerEnvironment(
                container_id=self.container.id,
                cwd="/testbed",
                timeout=120,
            ),
            **agent_config,
        )

        try:
            # We wrap in try/except to ensure we capture trace even if it crashes
            # Build kwargs for template variables
            run_kwargs = {}
            
            # Pass skills if the config has the {{ skills }} placeholder
            if "{{ skills }}" in agent_config.get("system_template", ""):
                run_kwargs["skills"] = skills
            
            # Pass system info if the config has these placeholders (original_mini.yaml)
            instance_template = agent_config.get("instance_template", "")
            if "{{system}}" in instance_template or "{{ system }}" in instance_template:
                run_kwargs["system"] = platform.system()
                run_kwargs["release"] = platform.release()
                run_kwargs["version"] = platform.version()
                run_kwargs["machine"] = platform.machine()
            
            result = agent.run(problem_statement, **run_kwargs)

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

            # Generate patch of changes from Docker container
            result = self.container.exec_run("git diff", workdir="/testbed")
            patch = result.output.decode() if result.output else ""

            # Explicit cleanup to prevent memory leaks
            del agent
            gc.collect()

            return patch, trace, metrics

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"  AGENT ERROR: {str(e)}")
            print(f"  Traceback:\n{error_trace}")
            gc.collect()  # Clean up even on error
            return "", f"Agent crashed: {str(e)}\n\nTraceback:\n{error_trace}", {"steps": 0, "estimated_tokens": 0, "num_messages": 0}

    def run_claude_code(self, problem_statement: str, skills: str, timeout: int = 600) -> Tuple[str, str, dict]:
        """Run Claude Code inside the Docker container.
        
        Uses `claude -p` (print mode) for non-interactive execution.
        Skills are injected via --append-system-prompt.
        
        Args:
            problem_statement: The problem/issue to solve
            skills: Skills string to inject into system prompt
            timeout: Max seconds for Claude Code to run
            
        Returns:
            patch: The git diff of changes made
            trace: Claude Code's output/reasoning
            metrics: Dict with execution metadata
        """
        import shlex
        
        # Get API key - required for Claude Code
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        
        # Build the claude command
        # Using -p for non-interactive (print) mode
        # --allowedTools to auto-approve common tools
        # --append-system-prompt to inject GEPA-learned skills
        # --output-format json for structured output
        
        # Escape the problem statement and skills for shell
        escaped_problem = problem_statement.replace("'", "'\\''")
        escaped_skills = skills.replace("'", "'\\''") if skills else ""
        
        # Build command parts
        cmd_parts = [
            "claude", "-p", f"'{escaped_problem}'",
            "--allowedTools", "'Bash,Read,Write,Edit,MultiEdit'",
            "--output-format", "json",
        ]
        
        # Add skills as system prompt if provided
        if escaped_skills:
            cmd_parts.extend(["--append-system-prompt", f"'{escaped_skills}'"])
        
        cmd = " ".join(cmd_parts)
        
        # Wrap in timeout and run
        full_cmd = f"timeout {timeout} {cmd}"
        
        print(f"    Running Claude Code (timeout={timeout}s)...")
        
        try:
            # Execute in container
            exec_result = self.container.exec_run(
                ["bash", "-c", full_cmd],
                workdir="/testbed",
                environment={
                    "ANTHROPIC_API_KEY": api_key,
                    "HOME": "/root",
                    "TERM": "xterm-256color",
                },
            )
            
            exit_code = exec_result.exit_code
            output = exec_result.output.decode() if exec_result.output else ""
            
            # Parse JSON output
            trace = ""
            session_id = None
            
            if exit_code == 124:
                # Timeout
                trace = f"Claude Code timed out after {timeout}s\n\nPartial output:\n{output}"
            elif exit_code != 0:
                trace = f"Claude Code exited with code {exit_code}\n\nOutput:\n{output}"
            else:
                # Try to parse JSON output
                try:
                    result_json = json.loads(output)
                    trace = result_json.get("result", output)
                    session_id = result_json.get("session_id")
                except json.JSONDecodeError:
                    # Not JSON, use raw output
                    trace = output
            
            # Get the git diff (patch)
            diff_result = self.container.exec_run("git diff", workdir="/testbed")
            patch = diff_result.output.decode() if diff_result.output else ""
            
            # Build metrics
            metrics = {
                "exit_code": exit_code,
                "session_id": session_id,
                "output_length": len(output),
                "timed_out": exit_code == 124,
            }
            
            return patch, trace, metrics
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"  CLAUDE CODE ERROR: {str(e)}")
            return "", f"Claude Code crashed: {str(e)}\n\nTraceback:\n{error_trace}", {
                "exit_code": -1,
                "session_id": None,
                "error": str(e),
            }

    def run(self, problem_statement: str, skills: str, model_name: str = "gpt-5-mini",
            config_path: str = None, timeout: int = 600) -> Tuple[str, str, dict]:
        """Run the configured agent type.
        
        Dispatches to run_agent() or run_claude_code() based on self.agent_type.
        
        Args:
            problem_statement: The problem to solve
            skills: Skills string to inject
            model_name: Model name (for minisweagent only)
            config_path: Config path (for minisweagent only)
            timeout: Timeout in seconds (for claude-code only)
            
        Returns:
            patch: The git diff of changes
            trace: Agent's conversation/output
            metrics: Execution metrics
        """
        if self.agent_type == "claude-code":
            return self.run_claude_code(problem_statement, skills, timeout=timeout)
        else:
            return self.run_agent(problem_statement, skills, model_name=model_name, config_path=config_path)

    def verify_with_patch(self, patch: str, f2p_only: bool = True, timeout: int = 300) -> Tuple[bool, str]:
        """Verify a patch using SWE-smith's run_patch_in_container.
        
        This is the proper way to evaluate patches - it:
        1. Creates a fresh container
        2. Checks out the correct commit with test files
        3. Applies the patch
        4. Runs the appropriate tests
        5. Cleans up
        
        Args:
            patch: The git diff patch to apply and test
            f2p_only: If True, only run FAIL_TO_PASS tests
            timeout: Test timeout in seconds
            
        Returns:
            (passed, test_output) tuple
        """
        if not self.current_task:
            return False, "No task set up"
        
        instance_id = self.current_task.get(KEY_INSTANCE_ID, "unknown")
        run_id = f"gepa_{uuid.uuid4().hex[:8]}"
        # IMPORTANT: Use RUN_EVALUATION_LOG_DIR to trigger is_eval=True in run_patch_in_container
        # This makes it do `git checkout HEAD~1` to restore test files
        log_dir = RUN_EVALUATION_LOG_DIR
        
        try:
            result = run_patch_in_container(
                instance=self.current_task,
                run_id=run_id,
                log_dir=log_dir,
                timeout=timeout,
                patch=patch if patch.strip() else None,
                commit=instance_id,  # Checkout the instance branch
                f2p_only=f2p_only,
                is_gold=False,  # We're testing a fix, not the gold solution
            )
            
            if result is None:
                return False, "run_patch_in_container returned None (error occurred)"
            
            logger, timed_out = result
            
            # Read the test output from the log file
            test_output_file = Path(log_dir) / run_id / instance_id / LOG_TEST_OUTPUT
            if test_output_file.exists():
                test_output = test_output_file.read_text()
            else:
                test_output = "Test output file not found"
            
            # Parse test results - check for failures in the output
            # SWE-smith uses pytest, so we look for standard pytest markers
            passed = not timed_out and "FAILED" not in test_output and "ERROR" not in test_output
            
            # Also check exit code from the log if available
            if "PASSED" in test_output or "passed" in test_output.lower():
                passed = True
            
            return passed, test_output
            
        except Exception as e:
            import traceback
            return False, f"Verification error: {e}\n{traceback.format_exc()}"

    def cleanup(self):
        """Cleanup Docker container after run."""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
                self.container = None
            except Exception as e:
                print(f"  WARNING: Failed to cleanup container: {e}")
