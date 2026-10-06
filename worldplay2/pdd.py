from __future__ import annotations

from dataclasses import dataclass

import torch

from .checkpoint import load_state_dict


@dataclass(frozen=True)
class FixedPDDBlock:
    expert: str
    local_index: int
    start: int
    end: int


class FixedPDD4Scheduler:
    """Fixed four-NFE consistency-PDD schedule used by Self-Forcing."""

    num_intervals = 128
    shift = 7.0
    boundary = 0.9

    def __init__(self):
        raw_boundary = self.boundary / (
            self.shift - (self.shift - 1.0) * self.boundary
        )
        high_count = int(round(self.num_intervals * (1.0 - raw_boundary)))
        high_count = min(max(high_count, 1), self.num_intervals - 1)
        raw = torch.cat([
            torch.linspace(1.0, raw_boundary, high_count + 1),
            torch.linspace(
                raw_boundary,
                0.0,
                self.num_intervals - high_count + 1,
            )[1:],
        ])
        self.sigmas = (
            self.shift * raw / (1.0 + (self.shift - 1.0) * raw)
        ).float()
        self.sigmas[0] = 1.0
        self.sigmas[high_count] = self.boundary
        self.sigmas[-1] = 0.0
        self.timesteps = self.sigmas * 1000.0
        self.boundary_index = high_count

        high_edges = self._partition(0, high_count)
        low_edges = self._partition(high_count, self.num_intervals)
        self._blocks = {
            "high": tuple(
                FixedPDDBlock("high", i, high_edges[i], high_edges[i + 1])
                for i in range(2)
            ),
            "low": tuple(
                FixedPDDBlock("low", i, low_edges[i], low_edges[i + 1])
                for i in range(2)
            ),
        }
        self.ordered_blocks = self._blocks["high"] + self._blocks["low"]

    @staticmethod
    def _partition(start: int, end: int) -> list[int]:
        return [start, start + (end - start) // 2, end]

    def blocks(self, expert: str) -> tuple[FixedPDDBlock, ...]:
        try:
            return self._blocks[expert]
        except KeyError as exc:
            raise ValueError(f"expert must be high/low, got {expert!r}") from exc

    def expert_range(self, expert: str) -> tuple[int, int]:
        if expert == "high":
            return 0, self.boundary_index
        if expert == "low":
            return self.boundary_index, self.num_intervals
        raise ValueError(f"expert must be high/low, got {expert!r}")

    def compact_head(
        self,
        expert: str,
        interval_weight: torch.Tensor,
        interval_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expert_start, expert_end = self.expert_range(expert)
        expected = expert_end - expert_start
        if interval_weight.ndim != 3 or interval_weight.shape[0] != expected:
            raise ValueError(
                f"{expert} PDD checkpoint has interval weight shape "
                f"{tuple(interval_weight.shape)}, expected [{expected}, O, I]"
            )
        if interval_bias.ndim != 2 or interval_bias.shape[0] != expected:
            raise ValueError(
                f"{expert} PDD checkpoint has interval bias shape "
                f"{tuple(interval_bias.shape)}, expected [{expected}, O]"
            )

        weights = []
        biases = []
        for block in self.blocks(expert):
            indices = [
                index - expert_start
                for index in range(block.start, block.end - 1)
            ]
            coefficients = (
                self.sigmas[block.start + 1:block.end]
                - self.sigmas[block.start:block.end - 1]
            ).to(interval_weight)
            displacement_weight = torch.einsum(
                "k,koi->oi", coefficients, interval_weight[indices]
            )
            displacement_bias = torch.einsum(
                "k,ko->o", coefficients, interval_bias[indices]
            )
            endpoint_index = block.end - 1 - expert_start
            weights.append(torch.cat(
                (displacement_weight, interval_weight[endpoint_index]), dim=0
            ))
            biases.append(torch.cat(
                (displacement_bias, interval_bias[endpoint_index]), dim=0
            ))
        return torch.stack(weights), torch.stack(biases)

    def prepare_checkpoint(
        self,
        path: str,
        *,
        expert: str,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Load either interval-head or already compact fixed-PDD weights."""
        state = load_state_dict(path)
        compact_weight = "head.block_weight"
        compact_bias = "head.block_bias"
        if compact_weight not in state or compact_bias not in state:
            weight_key = next((
                key for key in (
                    "head.weight",
                    "head.pdd_weight",
                    "head.head.weight",
                )
                if key in state and state[key].ndim == 3
            ), None)
            bias_key = next((
                key for key in (
                    "head.bias",
                    "head.pdd_bias",
                    "head.head.bias",
                )
                if key in state and state[key].ndim == 2
            ), None)
            if weight_key is None or bias_key is None:
                raise KeyError(
                    f"{path} has neither compact fixed-PDD heads nor an "
                    "interval PDD head bank"
                )
            block_weight, block_bias = self.compact_head(
                expert, state.pop(weight_key), state.pop(bias_key)
            )
            state[compact_weight] = block_weight
            state[compact_bias] = block_bias

        # A compact resume can retain a now-unused standard projection.
        state.pop("head.head.weight", None)
        state.pop("head.head.bias", None)
        for key, tensor in state.items():
            if tensor.is_floating_point() and tensor.dtype != dtype:
                state[key] = tensor.to(dtype=dtype)
        return state
