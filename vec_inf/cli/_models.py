"""Data models for CLI rendering."""

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
