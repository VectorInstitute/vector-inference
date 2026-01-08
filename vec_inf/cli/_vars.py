"""Constants for CLI rendering.

This module defines mappings for model type priorities, colors, and engine name mappings
used in the CLI display formatting.
"""

from typing import get_args

from vec_inf.client._slurm_vars import MODEL_TYPES


# Extract model type values from the Literal type
_MODEL_TYPES = get_args(MODEL_TYPES)

# Rich color options (prioritizing current colors, with fallbacks for additional types)
_RICH_COLORS = [
    "cyan",
    "bright_blue",
    "purple",
    "bright_magenta",
    "green",
    "yellow",
    "bright_green",
    "bright_yellow",
    "red",
    "bright_red",
    "blue",
    "magenta",
    "bright_cyan",
    "white",
    "bright_white",
]

# Mapping of model types to their display priority (lower numbers shown first)
MODEL_TYPE_PRIORITY = {model_type: idx for idx, model_type in enumerate(_MODEL_TYPES)}

# Mapping of model types to their display colors in Rich
MODEL_TYPE_COLORS = {
    model_type: _RICH_COLORS[idx % len(_RICH_COLORS)]
    for idx, model_type in enumerate(_MODEL_TYPES)
}

# Inference engine choice and name mapping
ENGINE_NAME_MAP = {
    "vllm": "vLLM",
    "sglang": "SGLang",
}
