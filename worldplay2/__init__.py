"""WorldPlay2 inference package.

Heavy CUDA/model modules are intentionally not imported here, so lightweight
utilities such as input and action parsing can be used without torchvision,
diffusers, or flash-attn.
"""

from .inputs import ActionSequence, parse_action_string

__all__ = ["ActionSequence", "parse_action_string"]
