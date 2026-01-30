import gc
import os
import yaml
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any

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
import docker


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
    def __init__(self):
        """Initialize harness with SWE-smith Docker containers."""
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
        # - Correct commit checked out
        # - Bug patch already applied
        # - All dependencies installed
        self.repo_profile = registry.get_from_inst(task_instance)
        self.current_task = task_instance
        self.container = self.repo_profile.get_container(task_instance)
        print(f"  Docker container created: {self.container.id[:12]}")
        
        # Debug: Verify container is running and accessible
        try:
            status = self.container.status
            result = self.container.exec_run("pwd", workdir="/testbed")
            pwd_output = result.output.decode().strip() if result.output else "N/A"
            print(f"  Container status: {status}, cwd: {pwd_output}")
        except Exception as e:
            print(f"  WARNING: Container verification failed: {e}")
        
    def run_agent(self, problem_statement: str, skills: str, model_name: str = "gemini/gemini-2.0-flash-exp") -> Tuple[str, str, dict]:
        """Run the agent in Docker container and return (patch, conversation_trace, metrics).

        The skills from GEPA are injected into the system template's {{ skills }} placeholder.
        This allows GEPA to evolve the agent's learned skills over time.

        Returns:
            patch: The git diff of changes made
            trace: Full conversation trace
            metrics: Dict with 'steps' (number of agent turns) and 'tokens' (estimated)
        """
        
        # Load our custom mini-swe-agent config from the project directory
        # This config has {{ skills }} placeholder that GEPA will optimize
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
            # Pass skills as a kwarg - matches {{ skills }} in system_template
            result = agent.run(problem_statement, skills=skills)

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

    def get_test_cmd(self, f2p_only: bool = False) -> str:
        """Get the proper test command from the repo profile.
        
        This uses SWE-smith's repo profile which knows how to:
        - Activate the correct conda environment
        - Run the right test framework (pytest, cargo test, npm test, etc.)
        - Pass the correct test files
        """
        if self.repo_profile and self.current_task:
            test_cmd, _ = self.repo_profile.get_test_cmd(self.current_task, f2p_only=f2p_only)
            return test_cmd
        # Fallback for Python repos
        return "source /opt/miniconda3/bin/activate; conda activate testbed; python -m pytest -v"

    def verify(self, test_cmd: str = None, f2p_only: bool = False) -> Tuple[bool, str]:
        """Run verification tests in Docker container and return (passed, output).

        Args:
            test_cmd: Optional custom test command. If None, uses repo profile's test command.
            f2p_only: If True and test_cmd is None, only run FAIL_TO_PASS tests.

        Returns both the pass/fail status and the full test output,
        which has info like:
        - Test failure messages
        - Stack traces
        - Compilation errors
        - Expected vs actual values
        """
        # Get test command from repo profile if not provided
        if test_cmd is None:
            test_cmd = self.get_test_cmd(f2p_only=f2p_only)
        
        # Run tests in Docker container using bash shell
        # This is needed because test_cmd may contain shell commands like 'source' and ';'
        result = self.container.exec_run(
            ["bash", "-c", test_cmd],
            workdir="/testbed",
            demux=True,  # Separate stdout/stderr
        )
        
        exit_code = result.exit_code
        stdout = result.output[0].decode() if result.output[0] else ""
        stderr = result.output[1].decode() if result.output[1] else ""
        
        test_output = f"=== TEST COMMAND: {test_cmd} ===\n"
        test_output += f"=== RETURN CODE: {exit_code} ===\n\n"
        
        if stdout:
            test_output += "=== STDOUT ===\n" + stdout + "\n"
        if stderr:
            test_output += "=== STDERR ===\n" + stderr + "\n"
        
        passed = exit_code == 0
        return passed, test_output

    def cleanup(self):
        """Cleanup Docker container after run."""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
                self.container = None
            except Exception as e:
                print(f"  WARNING: Failed to cleanup container: {e}")
