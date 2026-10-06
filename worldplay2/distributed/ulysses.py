from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class UlyssesSplitPlan:
    global_sequence: int
    num_heads: int
    world_size: int
    rank: int
    sequence_splits: tuple[int, ...]
    sequence_start: int
    local_sequence: int
    local_heads: int


def _balanced_splits(total: int, parts: int) -> tuple[int, ...]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total < parts:
        raise ValueError(
            f"sequence length {total} must be >= world size {parts}"
        )
    base, remainder = divmod(total, parts)
    return tuple(
        base + (1 if rank < remainder else 0)
        for rank in range(parts)
    )


@lru_cache(maxsize=256)
def get_ulysses_split_plan(
    global_sequence: int,
    num_heads: int,
    world_size: int,
    rank: int,
) -> UlyssesSplitPlan:
    if num_heads % world_size:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by world_size={world_size}"
        )
    splits = _balanced_splits(global_sequence, world_size)
    return UlyssesSplitPlan(
        global_sequence=global_sequence,
        num_heads=num_heads,
        world_size=world_size,
        rank=rank,
        sequence_splits=splits,
        sequence_start=sum(splits[:rank]),
        local_sequence=splits[rank],
        local_heads=num_heads // world_size,
    )


_RECEIVE_BUFFERS: dict[tuple, torch.Tensor] = {}
_STATS = {
    "packed_qkv_calls": 0,
    "output_calls": 0,
    "final_gather_calls": 0,
    "qkv_collectives_eliminated": 0,
    "buffer_hits": 0,
    "buffer_misses": 0,
    "bytes_exchanged": 0,
}


def _receive_buffer(name: str, reference: torch.Tensor, elements: int):
    key = (
        name,
        reference.device.type,
        reference.device.index,
        reference.dtype,
    )
    buffer = _RECEIVE_BUFFERS.get(key)
    if buffer is None or buffer.numel() < elements:
        buffer = torch.empty(
            elements,
            dtype=reference.dtype,
            device=reference.device,
        )
        _RECEIVE_BUFFERS[key] = buffer
        _STATS["buffer_misses"] += 1
    else:
        _STATS["buffer_hits"] += 1
    return buffer[:elements]


def shard_sequence(
    tensor: torch.Tensor | None,
    *,
    global_sequence: int,
    rank: int,
    world_size: int,
    dim: int,
) -> torch.Tensor | None:
    if tensor is None or world_size <= 1:
        return tensor
    splits = _balanced_splits(global_sequence, world_size)
    start = sum(splits[:rank])
    return tensor.narrow(dim, start, splits[rank]).contiguous()


def _pack_qkv_for_destinations(q, k, v, world_size, local_heads):
    batch, local_sequence, _, head_dim = q.shape

    def destination_major(tensor):
        return tensor.contiguous().view(
            batch,
            local_sequence,
            world_size,
            local_heads,
            head_dim,
        ).permute(2, 0, 1, 3, 4)

    return torch.stack(
        [destination_major(q), destination_major(k), destination_major(v)],
        dim=1,
    ).contiguous()


def _unpack_received_qkv(
    output_flat,
    *,
    sequence_splits,
    batch,
    local_heads,
    head_dim,
):
    source_chunks = []
    offset = 0
    for source_sequence in sequence_splits:
        elements = (
            3 * batch * source_sequence * local_heads * head_dim
        )
        source_chunks.append(
            output_flat[offset:offset + elements].view(
                3,
                batch,
                source_sequence,
                local_heads,
                head_dim,
            )
        )
        offset += elements
    return torch.cat(source_chunks, dim=2).contiguous()


def _pack_output_for_destinations(output, sequence_splits):
    return torch.cat([
        chunk.contiguous().view(-1)
        for chunk in torch.split(output, sequence_splits, dim=1)
    ])


def _unpack_received_output(
    output_flat,
    *,
    world_size,
    batch,
    local_sequence,
    local_heads,
    num_heads,
    head_dim,
):
    return output_flat.view(
        world_size,
        batch,
        local_sequence,
        local_heads,
        head_dim,
    ).permute(1, 2, 0, 3, 4).reshape(
        batch,
        local_sequence,
        num_heads,
        head_dim,
    ).contiguous()


def packed_qkv_all_to_all(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    global_sequence: int,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather sequence and shard heads with one packed all-to-all-single."""
    if not dist.is_initialized():
        raise RuntimeError("packed QKV all-to-all requires distributed init")
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return q, k, v
    rank = dist.get_rank(group)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, and V must have identical local shapes")

    batch, local_sequence, num_heads, head_dim = q.shape
    plan = get_ulysses_split_plan(
        global_sequence, num_heads, world_size, rank)
    if local_sequence != plan.local_sequence:
        raise ValueError(
            f"rank {rank} expected {plan.local_sequence} local tokens, "
            f"got {local_sequence}"
        )

    # [destination, qkv, batch, local_sequence, local_heads, head_dim]
    packed = _pack_qkv_for_destinations(
        q, k, v, world_size, plan.local_heads)
    input_flat = packed.view(-1)
    per_destination = (
        3 * batch * local_sequence * plan.local_heads * head_dim
    )
    input_splits = [per_destination] * world_size
    output_splits = [
        3 * batch * seq * plan.local_heads * head_dim
        for seq in plan.sequence_splits
    ]
    output_flat = _receive_buffer(
        "packed_qkv", input_flat, sum(output_splits))
    dist.all_to_all_single(
        output_flat,
        input_flat,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    packed_global = _unpack_received_qkv(
        output_flat,
        sequence_splits=plan.sequence_splits,
        batch=batch,
        local_heads=plan.local_heads,
        head_dim=head_dim,
    )

    _STATS["packed_qkv_calls"] += 1
    _STATS["qkv_collectives_eliminated"] += 2
    _STATS["bytes_exchanged"] += (
        input_flat.numel() * input_flat.element_size())
    return tuple(packed_global.unbind(0))


def output_all_to_all(
    output: torch.Tensor,
    *,
    global_sequence: int,
    num_heads: int,
    group=None,
) -> torch.Tensor:
    """Shard sequence and gather heads after attention."""
    if not dist.is_initialized():
        raise RuntimeError("output all-to-all requires distributed init")
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return output
    rank = dist.get_rank(group)

    batch, sequence, local_heads, head_dim = output.shape
    plan = get_ulysses_split_plan(
        global_sequence, num_heads, world_size, rank)
    if sequence != global_sequence or local_heads != plan.local_heads:
        raise ValueError(
            "attention output shape does not match the Ulysses split plan: "
            f"shape={tuple(output.shape)}, plan={plan}"
        )

    input_flat = _pack_output_for_destinations(
        output, plan.sequence_splits)
    input_splits = [
        batch * seq * local_heads * head_dim
        for seq in plan.sequence_splits
    ]
    per_source = (
        batch * plan.local_sequence * local_heads * head_dim
    )
    output_splits = [per_source] * world_size
    output_flat = _receive_buffer(
        "attention_output", input_flat, sum(output_splits))
    dist.all_to_all_single(
        output_flat,
        input_flat,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    local = _unpack_received_output(
        output_flat,
        world_size=world_size,
        batch=batch,
        local_sequence=plan.local_sequence,
        local_heads=local_heads,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    _STATS["output_calls"] += 1
    _STATS["bytes_exchanged"] += (
        input_flat.numel() * input_flat.element_size())
    return local


def gather_sequence(
    tensor: torch.Tensor,
    *,
    global_sequence: int,
    group=None,
) -> torch.Tensor:
    """Gather a known balanced sequence split without object collectives."""
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    rank = dist.get_rank(group)
    splits = _balanced_splits(global_sequence, world_size)
    if tensor.shape[1] != splits[rank]:
        raise ValueError(
            f"rank {rank} expected {splits[rank]} tokens, got {tensor.shape[1]}"
        )

    max_sequence = max(splits)
    padded_shape = list(tensor.shape)
    padded_shape[1] = max_sequence
    padded = tensor.new_zeros(padded_shape)
    padded[:, :tensor.shape[1]].copy_(tensor)
    padded_flat = padded.contiguous().view(-1)
    gathered_flat = _receive_buffer(
        "final_sequence_gather",
        padded_flat,
        world_size * padded_flat.numel(),
    )
    dist.all_gather_into_tensor(
        gathered_flat,
        padded_flat,
        group=group,
    )
    gathered = gathered_flat.view(world_size, *padded.shape)
    result = torch.cat(
        [chunk[:, :length] for chunk, length in zip(gathered, splits)],
        dim=1,
    ).contiguous()
    _STATS["final_gather_calls"] += 1
    _STATS["bytes_exchanged"] += (
        padded.numel() * padded.element_size())
    return result


def get_ulysses_stats() -> dict:
    return dict(_STATS)


def clear_ulysses_buffers() -> None:
    _RECEIVE_BUFFERS.clear()
