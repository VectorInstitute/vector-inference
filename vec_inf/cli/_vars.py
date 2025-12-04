"""Constants for CLI rendering.

This module defines mappings for model type priorities, colors, and engine name mappings
used in the CLI display formatting.
"""

# Mapping of model types to their display priority (lower numbers shown first)
MODEL_TYPE_PRIORITY = {
    "LLM": 0,
    "VLM": 1,
    "Text_Embedding": 2,
    "Reward_Modeling": 3,
}

# Mapping of model types to their display colors in Rich
MODEL_TYPE_COLORS = {
    "LLM": "cyan",
    "VLM": "bright_blue",
    "Text_Embedding": "purple",
    "Reward_Modeling": "bright_magenta",
}

# Inference engine choice and name mapping
ENGINE_NAME_MAP = {
    "vllm": "vLLM",
    "sglang": "SGLang",
}
