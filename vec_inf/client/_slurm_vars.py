"""Slurm cluster configuration variables."""

import os
import warnings
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from typing_extensions import Literal


CACHED_CONFIG_DIR = Path("/model-weights/vec-inf-shared")


def load_env_config() -> dict[str, Any]:
    """Load the environment configuration."""

    def load_yaml_config(path: Path) -> dict[str, Any]:
        """Load YAML config with error handling."""
        try:
            with path.open() as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Could not find config: {path}") from err
        except yaml.YAMLError as err:
            raise ValueError(f"Error parsing YAML config at {path}: {err}") from err

    cached_config_path = CACHED_CONFIG_DIR / "environment.yaml"
    default_path = (
        cached_config_path
        if cached_config_path.exists()
        else Path(__file__).resolve().parent.parent / "config" / "environment.yaml"
    )
    config = load_yaml_config(default_path)

    user_path = os.getenv("VEC_INF_CONFIG_DIR")
    if user_path:
        user_path_obj = Path(user_path, "environment.yaml")
        if user_path_obj.exists():
            user_config = load_yaml_config(user_path_obj)
            config.update(user_config)
        else:
            warnings.warn(
                f"WARNING: Could not find user config directory: {user_path}, revert to default config located at {default_path}",
                UserWarning,
                stacklevel=2,
            )

    return config


_config = load_env_config()

# Extract path values
IMAGE_PATH = {
    "vllm": _config["paths"]["vllm_image_path"],
    "sglang": _config["paths"]["sglang_image_path"],
}
CACHED_MODEL_CONFIG_PATH = Path(_config["paths"]["cached_model_config_path"])

# Extract containerization info
CONTAINER_LOAD_CMD = _config["containerization"]["module_load_cmd"]
CONTAINER_MODULE_NAME = _config["containerization"]["module_name"]

# Extract limits
MAX_GPUS_PER_NODE = _config["limits"]["max_gpus_per_node"]
MAX_NUM_NODES = _config["limits"]["max_num_nodes"]
MAX_CPUS_PER_TASK = _config["limits"]["max_cpus_per_task"]


# Create dynamic Literal types
def create_literal_type(values: list[str], fallback: str = "") -> Any:
    """Create a Literal type from a list, with configurable fallback."""
    if not values:
        return Literal[fallback]
    return Literal[tuple(values)]


QOS: TypeAlias = create_literal_type(_config["allowed_values"]["qos"])  # type: ignore[valid-type]
PARTITION: TypeAlias = create_literal_type(_config["allowed_values"]["partition"])  # type: ignore[valid-type]
RESOURCE_TYPE: TypeAlias = create_literal_type(  # type: ignore[valid-type]
    _config["allowed_values"]["resource_type"]
)

# Model types available derived from the cached model config
MODEL_TYPES: TypeAlias = create_literal_type(_config["model_types"])  # type: ignore[valid-type]

# Required arguments for launching jobs and corresponding environment variables
REQUIRED_ARGS: dict[str, str | None] = _config["required_args"]

# Running sglang requires python version
PYTHON_VERSION: str = _config["python_version"]

# Extract default arguments
DEFAULT_ARGS: dict[str, str] = _config["default_args"]
