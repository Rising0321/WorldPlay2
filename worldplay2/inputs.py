from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from worldplay2.distributed.ops import init_distributed_group


_MOVEMENT = {
    "w": (1, 0),
    "s": (-1, 0),
    "a": (0, -1),
    "d": (0, 1),
    "wa": (1, -1),
    "wd": (1, 1),
    "sa": (-1, -1),
    "sd": (-1, 1),
}
_ROTATION = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class ActionSequence:
    tensor: torch.Tensor
    source: str
    perspective: str

    @property
    def latent_frames(self) -> int:
        return int(self.tensor.shape[0])


@dataclass(frozen=True)
class GenerationInput:
    image: Path
    actions: ActionSequence
    prompts: dict[str, str]
    chunk_prompts: tuple[str, ...]
    output_name: str


@dataclass(frozen=True)
class InferenceInputs:
    items: list[GenerationInput]
    mode: str
    sample_shift: float
    sample_steps: int
    guide_scale: float | tuple[float, ...]


@dataclass(frozen=True)
class RuntimeContext:
    rank: int
    world_size: int
    local_rank: int
    output_dir: Path
    offload_model: bool
    seed: int

    def finish(self) -> None:
        if dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])
            dist.destroy_process_group()


def parse_action_string(
    value: str,
    *,
    perspective: str = "tps",
    rotation_speed_deg: float = 3.0,
    yaw_rotation_speed_deg: float | None = None,
    pitch_rotation_speed_deg: float | None = None,
) -> ActionSequence:
    """Convert ``w+right-8,s-4`` directly to model actions.

    Command durations sum to the total latent length. The first latent is
    always stationary, so the first command applies for ``duration - 1``
    subsequent latents.

    Yaw (left/right) and pitch (up/down) use independent degrees per latent.
    Each unspecified axis falls back to ``rotation_speed_deg``.
    """
    if perspective not in {"tps", "fps"}:
        raise ValueError("perspective must be 'tps' or 'fps'")
    if yaw_rotation_speed_deg is None:
        yaw_rotation_speed_deg = rotation_speed_deg
    if pitch_rotation_speed_deg is None:
        pitch_rotation_speed_deg = rotation_speed_deg
    for name, speed in (
        ("yaw_rotation_speed_deg", yaw_rotation_speed_deg),
        ("pitch_rotation_speed_deg", pitch_rotation_speed_deg),
    ):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if not value or not value.strip():
        raise ValueError("action string cannot be empty")

    perspective_value = 0.0 if perspective == "tps" else 1.0
    rows: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, perspective_value, 0.0]]
    first_command = True

    for raw_command in value.split(","):
        command = raw_command.strip()
        if not command:
            continue
        action_text, separator, duration_text = command.rpartition("-")
        if not separator or not action_text.strip():
            raise ValueError(
                f"invalid action command {command!r}; expected action-duration"
            )
        try:
            duration = int(duration_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"duration must be a positive integer in {command!r}"
            ) from exc
        if duration <= 0 or str(duration) != duration_text.strip():
            raise ValueError(
                f"duration must be a positive integer in {command!r}"
            )

        pitch = yaw = forward = lateral = space = 0.0
        tokens = [token.strip().lower() for token in action_text.split("+")]
        if not all(tokens):
            raise ValueError(f"empty action token in {command!r}")
        seen_categories: set[str] = set()
        for token in tokens:
            if token in _MOVEMENT:
                if "movement" in seen_categories:
                    raise ValueError(f"multiple movement tokens in {command!r}")
                seen_categories.add("movement")
                forward, lateral = map(float, _MOVEMENT[token])
            elif token in _ROTATION:
                if token in {"up", "down"}:
                    if "pitch" in seen_categories:
                        raise ValueError(f"multiple pitch tokens in {command!r}")
                    seen_categories.add("pitch")
                else:
                    if "yaw" in seen_categories:
                        raise ValueError(f"multiple yaw tokens in {command!r}")
                    seen_categories.add("yaw")
                pitch_sign, yaw_sign = _ROTATION[token]
                pitch += pitch_sign * pitch_rotation_speed_deg
                yaw += yaw_sign * yaw_rotation_speed_deg
            elif token == "space":
                if "space" in seen_categories:
                    raise ValueError(f"duplicate space token in {command!r}")
                seen_categories.add("space")
                space = 1.0
            elif token == "none":
                if len(tokens) != 1:
                    raise ValueError(
                        f"'none' cannot be combined with other actions "
                        f"in {command!r}"
                    )
            else:
                supported = sorted(
                    set(_MOVEMENT) | set(_ROTATION) | {"none", "space"}
                )
                raise ValueError(
                    f"unknown action token {token!r}; supported: {supported}"
                )

        row = [pitch, yaw, forward, lateral, perspective_value, space]
        repeat_count = duration - 1 if first_command else duration
        rows.extend([row.copy() for _ in range(repeat_count)])
        first_command = False

    if first_command:
        raise ValueError("action string did not contain any commands")
    return ActionSequence(
        tensor=torch.tensor(rows, dtype=torch.float32),
        source=value,
        perspective=perspective,
    )


def _parse_prompt_event(
    *,
    value: str,
    prompts: dict[str, str],
    latent_frames: int,
    chunk_length: int,
) -> tuple[str, ...]:
    """Parse ``prompt1-8,prompt2-8`` into one prompt name per chunk."""
    if not value or not value.strip():
        raise ValueError("prompt_event cannot be empty")

    latent_prompts: list[str] = []
    cumulative = 0
    for raw_event in value.split(","):
        event = raw_event.strip()
        prompt_name, separator, duration_text = event.rpartition("-")
        prompt_name = prompt_name.strip()
        if not separator or not prompt_name:
            raise ValueError(
                f"invalid prompt event {event!r}; expected prompt_name-duration"
            )
        if prompt_name not in prompts:
            raise ValueError(
                f"prompt_event references missing prompt {prompt_name!r}"
            )
        try:
            duration = int(duration_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"prompt duration must be a positive integer in {event!r}"
            ) from exc
        if duration <= 0 or str(duration) != duration_text.strip():
            raise ValueError(
                f"prompt duration must be a positive integer in {event!r}"
            )

        cumulative += duration
        if cumulative < latent_frames and cumulative % chunk_length:
            raise ValueError(
                "prompt_event boundaries must align with chunk_length; "
                f"boundary {cumulative} is not divisible by {chunk_length}"
            )
        latent_prompts.extend([prompt_name] * duration)

    if len(latent_prompts) != latent_frames:
        raise ValueError(
            f"prompt_event covers {len(latent_prompts)} latent frames, "
            f"but action produces {latent_frames}"
        )
    return tuple(
        latent_prompts[index]
        for index in range(0, latent_frames, chunk_length)
    )


def load_generation_inputs(
    *,
    input_json: str,
    rotation_speed_deg: float = 3.0,
    yaw_rotation_speed_deg: float | None = None,
    pitch_rotation_speed_deg: float | None = None,
    chunk_length: int,
    sink_size: int,
    temporal_size: int,
) -> list[GenerationInput]:
    json_path = Path(input_json).expanduser()
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("input JSON must contain a non-empty list")

    items = []
    for index, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict):
            raise ValueError(f"input JSON item {index} must be an object")
        image_path = raw_item.get("image_path")
        action = raw_item.get("action")
        prompt_event = raw_item.get("prompt_event")
        perspective = raw_item.get("perspective")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f"input JSON item {index} requires image_path")
        if not isinstance(action, str):
            raise ValueError(f"input JSON item {index} requires action")
        if not isinstance(prompt_event, str):
            raise ValueError(f"input JSON item {index} requires prompt_event")
        if perspective not in {"tps", "fps"}:
            raise ValueError(
                f"input JSON item {index} perspective must be 'tps' or 'fps'"
            )

        prompts = {
            key: value
            for key, value in raw_item.items()
            if key.startswith("prompt")
            and key != "prompt_event"
            and isinstance(value, str)
            and value.strip()
        }
        if not prompts:
            raise ValueError(
                f"input JSON item {index} requires at least one prompt field"
            )
        actions = load_action_sequence(
            action=action,
            perspective=perspective,
            rotation_speed_deg=rotation_speed_deg,
            yaw_rotation_speed_deg=yaw_rotation_speed_deg,
            pitch_rotation_speed_deg=pitch_rotation_speed_deg,
            chunk_length=chunk_length,
            sink_size=sink_size,
            temporal_size=temporal_size,
        )
        chunk_prompts = _parse_prompt_event(
            value=prompt_event,
            prompts=prompts,
            latent_frames=actions.latent_frames,
            chunk_length=chunk_length,
        )
        image = Path(image_path).expanduser()
        if not image.is_absolute():
            image = json_path.parent / image
        output_name = str(
            raw_item.get("output_name") or f"{index:04d}_{image.stem}"
        ).strip()
        items.append(
            GenerationInput(
                image=image,
                actions=actions,
                prompts=prompts,
                chunk_prompts=chunk_prompts,
                output_name=output_name,
            )
        )
    return items


def validate_geometry(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height % 64 or width % 64:
        raise ValueError(
            "height and width must be divisible by 64 for HR/LR VAE and patch alignment"
        )


def load_action_sequence(
    *,
    action: str,
    perspective: str,
    rotation_speed_deg: float = 3.0,
    yaw_rotation_speed_deg: float | None = None,
    pitch_rotation_speed_deg: float | None = None,
    chunk_length: int,
    sink_size: int,
    temporal_size: int,
) -> ActionSequence:
    """Parse an action string and validate its generation layout.

    This validates both the action syntax and its relationships with the
    chunked stage-three inference inputs.
    """
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if chunk_length % 2:
        raise ValueError(
            "chunk_length must be even for HR-to-LR temporal downsampling"
        )
    if sink_size < 0:
        raise ValueError("sink_size must be non-negative")
    if sink_size > chunk_length:
        raise ValueError("sink_size must not exceed chunk_length")
    if not 1 <= temporal_size <= chunk_length:
        raise ValueError("temporal_size must be in [1, chunk_length]")

    actions = parse_action_string(
        action,
        perspective=perspective,
        rotation_speed_deg=rotation_speed_deg,
        yaw_rotation_speed_deg=yaw_rotation_speed_deg,
        pitch_rotation_speed_deg=pitch_rotation_speed_deg,
    )
    if actions.latent_frames % chunk_length:
        raise ValueError(
            f"action produces {actions.latent_frames} latent frames, which is "
            f"not divisible by chunk_length={chunk_length}"
        )
    if sink_size >= actions.latent_frames:
        raise ValueError("sink_size must be smaller than the action length")
    return actions


def load_inference_inputs(
    args: Any,
    *,
    default_guide_scale: float | tuple[float, ...],
) -> InferenceInputs:
    """Resolve and validate all data and sampling inputs from CLI arguments."""
    validate_geometry(args.height, args.width)
    items = load_generation_inputs(
        input_json=args.input_json,
        rotation_speed_deg=args.rotation_speed_deg,
        yaw_rotation_speed_deg=getattr(args, "yaw_rotation_speed_deg", None),
        pitch_rotation_speed_deg=getattr(args, "pitch_rotation_speed_deg", None),
        chunk_length=args.chunk_length,
        sink_size=args.sink_size,
        temporal_size=args.temporal_size,
    )

    mode = str(args.inference_mode)
    if mode not in {"bi", "ar", "pdd"}:
        raise ValueError(f"unsupported inference mode: {mode!r}")
    sample_shift = (
        float(args.sample_shift)
        if args.sample_shift is not None
        else (7.0 if mode == "pdd" else 5.0)
    )
    if not math.isfinite(sample_shift) or sample_shift <= 0:
        raise ValueError("sample_shift must be a positive finite number")
    sample_steps = (
        int(args.sample_steps)
        if args.sample_steps is not None
        else (4 if mode == "pdd" else 40)
    )
    if sample_steps <= 0:
        raise ValueError("sample_steps must be positive")
    guide_scale = (
        args.sample_guide_scale
        if args.sample_guide_scale is not None
        else (1.0 if mode == "pdd" else default_guide_scale)
    )
    guide_scales = (
        (float(guide_scale),)
        if isinstance(guide_scale, (int, float))
        else tuple(float(scale) for scale in guide_scale)
    )
    if not guide_scales or any(
        not math.isfinite(scale) or scale < 0 for scale in guide_scales
    ):
        raise ValueError(
            "sample_guide_scale must contain finite non-negative values"
        )

    if mode == "pdd":
        if sample_steps != 4:
            raise ValueError("--inference_mode pdd requires --sample_steps 4")
        if not math.isclose(sample_shift, 7.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("--inference_mode pdd requires --sample_shift 7")
        if any(
            not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-8)
            for scale in guide_scales
        ):
            raise ValueError(
                "--inference_mode pdd requires --sample_guide_scale 1"
            )

    return InferenceInputs(
        items=items,
        mode=mode,
        sample_shift=sample_shift,
        sample_steps=sample_steps,
        guide_scale=guide_scale,
    )


def initialize_runtime(args: Any) -> RuntimeContext:
    """Validate runtime options and initialize distributed execution."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be in [0, WORLD_SIZE)")
    if not 0 <= local_rank < world_size:
        raise ValueError("LOCAL_RANK must be in [0, WORLD_SIZE)")

    if world_size == 1 and (
        args.t5_fsdp or args.dit_fsdp or args.ulysses_size != 1
    ):
        raise ValueError(
            "FSDP/SP options require torchrun with multiple processes"
        )

    if args.ulysses_size <= 0:
        raise ValueError("ulysses_size must be positive")
    if args.ulysses_size > 1 and args.ulysses_size != world_size:
        raise ValueError("ulysses_size must equal world size")

    fp8_enabled = bool(args.fp8_ffn or args.fp8_qkv)
    if fp8_enabled and args.dit_fsdp:
        raise ValueError("DiT FP8 cannot be combined with --dit_fsdp")
    if args.fp8_qkv and args.disable_self_qkv_fusion:
        raise ValueError(
            "--fp8_qkv requires self-QKV fusion; remove "
            "--disable_self_qkv_fusion"
        )
    if args.fp8_ffn and not (
        0 <= args.fp8_start_block <= args.fp8_end_block < 40
    ):
        raise ValueError(
            "FP8 block range must satisfy 0 <= start <= end < 40"
        )
    if args.fp8_qkv and not (
        0 <= args.fp8_qkv_start_block <= args.fp8_qkv_end_block < 40
    ):
        raise ValueError(
            "FP8 QKV block range must satisfy 0 <= start <= end < 40"
        )

    offload_model = args.offload_model
    if offload_model is None:
        offload_model = world_size == 1
    if fp8_enabled and offload_model:
        raise ValueError("DiT FP8 requires --offload_model false")

    output_dir = Path(args.output_dir).expanduser()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not (
        args.disable_compiled_qk_rope
        and args.disable_compiled_block_glue
        and args.disable_compiled_ffn
    ):
        compile_root = (
            Path(args.compile_cache_dir).expanduser()
            / f"rank_{rank:02d}"
        )
        inductor_cache = compile_root / "torchinductor"
        triton_cache = compile_root / "triton"
        for cache_dir in (inductor_cache, triton_cache):
            cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
        if args.ulysses_size > 1:
            init_distributed_group()
        dist.barrier(device_ids=[local_rank])

    seed = (
        int(args.seed)
        if args.seed >= 0
        else random.randint(0, sys.maxsize)
    )
    if dist.is_initialized():
        seed_box = [seed if rank == 0 else None]
        dist.broadcast_object_list(seed_box, src=0)
        seed = int(seed_box[0])

    return RuntimeContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        output_dir=output_dir,
        offload_model=bool(offload_model),
        seed=seed,
    )
