from __future__ import annotations

import logging
from functools import lru_cache

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_QK_ROPE_STATS = {
    "compiled_calls": 0,
    "eager_calls": 0,
    "compile_failures": 0,
}
_REGION_STATS = {
    "layer_norm_modulation": {
        "triton_calls": 0,
        "eager_calls": 0,
        "failures": 0,
    },
    "modulation": {"compiled_calls": 0, "eager_calls": 0, "failures": 0},
    "gated_residual": {
        "compiled_calls": 0,
        "eager_calls": 0,
        "failures": 0,
    },
    "action_mlp": {
        "compiled_calls": 0,
        "eager_calls": 0,
        "failures": 0,
    },
    "ffn": {"compiled_calls": 0, "eager_calls": 0, "failures": 0},
}
_DISABLED_REGIONS: set[str] = set()


if triton is not None:

    @triton.jit
    def _layer_norm_modulation_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        output_ptr,
        sequence_length: tl.constexpr,
        hidden_size: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        scale_stride_b: tl.constexpr,
        scale_stride_s: tl.constexpr,
        shift_stride_b: tl.constexpr,
        shift_stride_s: tl.constexpr,
        eps: tl.constexpr,
        INPUT_DTYPE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One Triton program normalizes and modulates one token row."""
        row = tl.program_id(0)
        batch_index = row // sequence_length
        sequence_index = row - batch_index * sequence_length
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < hidden_size

        x_offsets = (
            batch_index * x_stride_b
            + sequence_index * x_stride_s
            + columns
        )
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / hidden_size
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / hidden_size
        normalized = centered * tl.rsqrt(variance + eps)
        # Match WorldPlay2LayerNorm exactly: FP32 normalization is rounded back to
        # the activation dtype before adaptive modulation evaluates in FP32.
        if INPUT_DTYPE == 1:
            normalized = normalized.to(tl.bfloat16).to(tl.float32)
        elif INPUT_DTYPE == 0:
            normalized = normalized.to(tl.float16).to(tl.float32)

        scale_offsets = (
            batch_index * scale_stride_b
            + sequence_index * scale_stride_s
            + columns
        )
        shift_offsets = (
            batch_index * shift_stride_b
            + sequence_index * shift_stride_s
            + columns
        )
        scale = tl.load(
            scale_ptr + scale_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        shift = tl.load(
            shift_ptr + shift_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        output = normalized * (1.0 + scale) + shift

        output_offsets = row * hidden_size + columns
        tl.store(output_ptr + output_offsets, output, mask=mask)


def qk_norm_rope_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_real: torch.Tensor,
    freqs_imag: torch.Tensor,
    *,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS-normalize Q/K and apply real-valued RoPE in FP32."""

    def normalize_and_rotate(x, weight):
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + eps
        )
        # Preserve the original WorldPlay2RMSNorm order: normalize in FP32, cast
        # back to projection dtype, then apply the learned norm weight.
        normalized = normalized.to(x.dtype) * weight
        batch, sequence, _ = normalized.shape
        pairs = normalized.float().view(
            batch, sequence, num_heads, head_dim // 2, 2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        cos = freqs_real.float().view(
            1, sequence, 1, head_dim // 2)
        sin = freqs_imag.float().view(
            1, sequence, 1, head_dim // 2)
        return torch.stack(
            [even * cos - odd * sin, even * sin + odd * cos],
            dim=-1,
        ).flatten(-2)

    return (
        normalize_and_rotate(q, norm_q_weight),
        normalize_and_rotate(k, norm_k_weight),
    )


@lru_cache(maxsize=16)
def _compiled_kernel(num_heads: int, head_dim: int, eps: float):
    def kernel(q, k, norm_q_weight, norm_k_weight, freqs_real, freqs_imag):
        return qk_norm_rope_eager(
            q,
            k,
            norm_q_weight,
            norm_k_weight,
            freqs_real,
            freqs_imag,
            eps=eps,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    return torch.compile(
        kernel,
        fullgraph=True,
        dynamic=True,
        mode="max-autotune-no-cudagraphs",
    )


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_flat: torch.Tensor,
    *,
    eps: float,
    num_heads: int,
    head_dim: int,
    compile_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Run compiled QK norm + RoPE, falling back to the same eager kernel."""
    args = (
        q,
        k,
        norm_q_weight,
        norm_k_weight,
        freqs_flat.real,
        freqs_flat.imag,
    )
    if compile_enabled:
        try:
            output = _compiled_kernel(num_heads, head_dim, eps)(*args)
            _QK_ROPE_STATS["compiled_calls"] += 1
            return output[0], output[1], True
        except Exception:
            _QK_ROPE_STATS["compile_failures"] += 1
            logging.exception(
                "Compiled QK norm + RoPE failed; using eager fallback")

    output = qk_norm_rope_eager(
        *args,
        eps=eps,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    _QK_ROPE_STATS["eager_calls"] += 1
    return output[0], output[1], False


def get_qk_rope_compile_stats() -> dict:
    return dict(_QK_ROPE_STATS)


# ---------------------------------------------------------------------------
# Repeated Wan block regions
# ---------------------------------------------------------------------------

def layer_norm_modulation_eager(
    hidden_states,
    scale,
    shift,
    *,
    eps,
):
    """Reference Wan LayerNorm followed by adaptive scale and shift."""
    normalized = F.layer_norm(
        hidden_states.float(),
        (hidden_states.shape[-1],),
        weight=None,
        bias=None,
        eps=eps,
    ).to(hidden_states.dtype)
    return normalized.float() * (1.0 + scale.float()) + shift.float()


def layer_norm_modulation(
    hidden_states,
    scale,
    shift,
    *,
    eps,
    enabled,
):
    """Fuse non-affine LayerNorm and Wan modulation in one Triton kernel."""
    name = "layer_norm_modulation"
    can_use_triton = (
        enabled
        and name not in _DISABLED_REGIONS
        and triton is not None
        and hidden_states.is_cuda
        and hidden_states.ndim == 3
        and scale.ndim == 3
        and shift.ndim == 3
        and hidden_states.shape == scale.shape == shift.shape
        and hidden_states.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        )
        and hidden_states.stride(-1) == 1
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
    )
    if can_use_triton:
        try:
            batch, sequence, hidden_size = hidden_states.shape
            output = torch.empty(
                hidden_states.shape,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            block_size = triton.next_power_of_2(hidden_size)
            if block_size > 65536:
                raise ValueError(
                    f"LayerNorm hidden size {hidden_size} is too large")
            num_warps = 8 if block_size >= 4096 else 4
            _layer_norm_modulation_kernel[(batch * sequence,)](
                hidden_states,
                scale,
                shift,
                output,
                sequence_length=sequence,
                hidden_size=hidden_size,
                x_stride_b=hidden_states.stride(0),
                x_stride_s=hidden_states.stride(1),
                scale_stride_b=scale.stride(0),
                scale_stride_s=scale.stride(1),
                shift_stride_b=shift.stride(0),
                shift_stride_s=shift.stride(1),
                eps=eps,
                INPUT_DTYPE=(
                    1 if hidden_states.dtype == torch.bfloat16
                    else 0 if hidden_states.dtype == torch.float16
                    else 2
                ),
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            _REGION_STATS[name]["triton_calls"] += 1
            return output, True
        except Exception:
            _REGION_STATS[name]["failures"] += 1
            _DISABLED_REGIONS.add(name)
            logging.exception(
                "Triton fused LayerNorm + modulation failed; "
                "disabling it and using eager fallback"
            )

    _REGION_STATS[name]["eager_calls"] += 1
    return layer_norm_modulation_eager(
        hidden_states,
        scale,
        shift,
        eps=eps,
    ), False


def modulation_eager(normalized, scale, shift):
    """Exact block modulation expression, evaluated in FP32."""
    return normalized.float() * (1.0 + scale.float()) + shift.float()


def gated_residual_eager(hidden_states, branch, gate):
    """Exact gated residual expression used by WorldPlay2AttentionBlock."""
    return hidden_states + branch * gate


def action_mlp_eager(
    hidden_states,
    action,
    input_weight,
    input_bias,
    output_weight,
    output_bias,
):
    """Action injection: concat, MLP, and residual add."""
    branch = torch.cat([hidden_states, action], dim=-1)
    branch = F.linear(branch, input_weight, input_bias)
    branch = F.silu(branch)
    branch = F.linear(branch, output_weight, output_bias)
    return hidden_states + branch


def ffn_eager(
    hidden_states,
    input_weight,
    input_bias,
    output_weight,
    output_bias,
):
    hidden_states = F.linear(
        hidden_states, input_weight, input_bias)
    hidden_states = F.gelu(hidden_states, approximate="tanh")
    return F.linear(hidden_states, output_weight, output_bias)


@lru_cache(maxsize=1)
def _compiled_modulation():
    return torch.compile(
        modulation_eager,
        fullgraph=True,
        dynamic=True,
        mode="max-autotune-no-cudagraphs",
    )


@lru_cache(maxsize=1)
def _compiled_gated_residual():
    return torch.compile(
        gated_residual_eager,
        fullgraph=True,
        dynamic=True,
        mode="max-autotune-no-cudagraphs",
    )


@lru_cache(maxsize=1)
def _compiled_action_mlp():
    return torch.compile(
        action_mlp_eager,
        fullgraph=True,
        dynamic=True,
        mode="max-autotune-no-cudagraphs",
    )


@lru_cache(maxsize=1)
def _compiled_ffn():
    return torch.compile(
        ffn_eager,
        fullgraph=True,
        dynamic=True,
        mode="max-autotune-no-cudagraphs",
    )


def _run_region(name, eager, compiled_factory, args, enabled):
    if enabled and name not in _DISABLED_REGIONS:
        try:
            output = compiled_factory()(*args)
            _REGION_STATS[name]["compiled_calls"] += 1
            return output, True
        except Exception:
            _REGION_STATS[name]["failures"] += 1
            _DISABLED_REGIONS.add(name)
            logging.exception(
                "Compiled region %s failed; disabling it and using eager",
                name,
            )
    _REGION_STATS[name]["eager_calls"] += 1
    return eager(*args), False


def modulation(normalized, scale, shift, *, enabled):
    return _run_region(
        "modulation",
        modulation_eager,
        _compiled_modulation,
        (normalized, scale, shift),
        enabled,
    )


def gated_residual(hidden_states, branch, gate, *, enabled):
    return _run_region(
        "gated_residual",
        gated_residual_eager,
        _compiled_gated_residual,
        (hidden_states, branch, gate),
        enabled,
    )


def action_mlp(
    hidden_states,
    action,
    input_weight,
    input_bias,
    output_weight,
    output_bias,
    *,
    enabled,
):
    return _run_region(
        "action_mlp",
        action_mlp_eager,
        _compiled_action_mlp,
        (
            hidden_states,
            action,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        ),
        enabled,
    )


def ffn(
    hidden_states,
    input_weight,
    input_bias,
    output_weight,
    output_bias,
    *,
    enabled,
):
    return _run_region(
        "ffn",
        ffn_eager,
        _compiled_ffn,
        (
            hidden_states,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        ),
        enabled,
    )


def get_compiled_region_stats() -> dict:
    return {
        name: dict(values)
        for name, values in _REGION_STATS.items()
    } | {
        "disabled_after_failure": sorted(_DISABLED_REGIONS),
    }
