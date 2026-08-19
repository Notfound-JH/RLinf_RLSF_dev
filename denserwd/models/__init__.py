try:
    from .octo import OctoReward
except Exception:
    OctoReward = None

try:
    from .qwen3_vl import Qwen3VLReward
except Exception:
    Qwen3VLReward = None

__all__ = ["TransformerReward", "OctoReward", "Qwen3VLReward"]
