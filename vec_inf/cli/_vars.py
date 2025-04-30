"""Constants for CLI rendering.

This module defines constant mappings for model type priorities and colors
used in the CLI display formatting.

Constants
---------
MODEL_TYPE_PRIORITY : dict
    Mapping of model types to their display priority (lower numbers shown first)

MODEL_TYPE_COLORS : dict
    Mapping of model types to their display colors in Rich

Notes
-----
These constants are used primarily by the ListCmdDisplay class to ensure
consistent sorting and color coding of different model types in the CLI output.
"""

MODEL_TYPE_PRIORITY = {
    "LLM": 0,
    "VLM": 1,
    "Text_Embedding": 2,
    "Reward_Modeling": 3,
}

MODEL_TYPE_COLORS = {
    "LLM": "cyan",
    "VLM": "bright_blue",
    "Text_Embedding": "purple",
    "Reward_Modeling": "bright_magenta",
}
