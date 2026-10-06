from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_state_dict(path: str, *, prefer_ema: bool = False) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = (
        load_file(str(checkpoint_path))
        if checkpoint_path.suffix == ".safetensors"
        else torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )

    if isinstance(state, dict) and ("generator" in state or "generator_ema" in state):
        key = "generator_ema" if prefer_ema and "generator_ema" in state else "generator"
        state = state[key]
    for wrapper in ("state_dict", "model_state_dict", "model"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]

    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint {checkpoint_path} does not contain a state dict")

    prefixes = ("module.", "model.")
    for prefix in prefixes:
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state


def load_expert_checkpoint(
    model,
    path: str,
    *,
    tag: str,
    prefer_ema: bool = False,
    strict: bool = True,
) -> None:
    state = load_state_dict(path, prefer_ema=prefer_ema)
    provided = set(state)
    required_prefixes = (
        "memory_compress.",
        "action_in_cont.",
        "action_in_disc.",
    )
    absent = [prefix for prefix in required_prefixes if not any(
        key.startswith(prefix) for key in provided
    )]
    if absent:
        raise ValueError(
            f"{tag} checkpoint is missing required parameter groups: {absent}"
        )
    missing, unexpected = model.load_state_dict(state, strict=strict)
    logging.info(
        "%s checkpoint: loaded=%d missing=%d unexpected=%d strict=%s",
        tag,
        len(state),
        len(missing),
        len(unexpected),
        strict,
    )
