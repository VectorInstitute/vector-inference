"""Slurm cluster configuration variables."""

import os
import warnings
from pathlib import Path
from typing import Any

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
LD_LIBRARY_PATH = _config["paths"]["ld_library_path"]
SINGULARITY_IMAGE = _config["paths"]["image_path"]
VLLM_NCCL_SO_PATH = _config["paths"]["vllm_nccl_so_path"]

# Extract containerization info
SINGULARITY_LOAD_CMD = _config["containerization"]["module_load_cmd"]
SINGULARITY_MODULE_NAME = _config["containerization"]["module_name"]

# Extract limits
MAX_GPUS_PER_NODE = _config["limits"]["max_gpus_per_node"]
MAX_NUM_NODES = _config["limits"]["max_num_nodes"]
MAX_CPUS_PER_TASK = _config["limits"]["max_cpus_per_task"]

# Create dynamic Literal types
QOS = Literal[tuple(_config["allowed_values"]["qos"])]  # type: ignore[valid-type]
PARTITION = Literal[tuple(_config["allowed_values"]["partition"])]  # type: ignore[valid-type]

# Extract default arguments
DEFAULT_ARGS = _config["default_args"]
