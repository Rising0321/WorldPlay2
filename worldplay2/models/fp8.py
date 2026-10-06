from __future__ import annotations

import gc
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_transformer_engine():
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
    except ImportError as exc:
        raise RuntimeError(
            "FFN FP8 requires NVIDIA Transformer Engine. Install the "
            "PyTorch extension with `pip install 'transformer_engine[pytorch]'`."
        ) from exc
    return te, DelayedScaling, Format


def make_fp8_recipe():
    """Hopper-friendly tensor-wise E4M3 delayed scaling for inference."""
    te, DelayedScaling, Format = _load_transformer_engine()
    available = te.is_fp8_available(return_reason=True)
    if isinstance(available, tuple):
        is_available, reason = available
    else:
        is_available, reason = available, "unsupported GPU/runtime"
    if not is_available:
        raise RuntimeError(f"Transformer Engine FP8 is unavailable: {reason}")
    return DelayedScaling(
        fp8_format=Format.E4M3,
        amax_history_len=16,
        amax_compute_algo="max",
        # Sequence-parallel ranks process different tokens. Local activation
        # scales avoid an extra amax all-reduce and are dequantized before the
        # next distributed attention.
        reduce_amax=False,
    )


def fp8_autocast(recipe, enabled: bool):
    if not enabled:
        return nullcontext()
    te, _, _ = _load_transformer_engine()
    autocast = getattr(te, "autocast", None)
    if autocast is None:
        autocast = te.fp8_autocast
    return autocast(enabled=True, recipe=recipe)


def _primary_fp8_init_context(te, recipe):
    """Create TE modules whose primary weights are stored only in FP8."""
    quantized_model_init = getattr(te, "quantized_model_init", None)
    if quantized_model_init is None:
        raise RuntimeError(
            "This FP8 inference path requires "
            "transformer_engine.pytorch.quantized_model_init. "
            "Transformer Engine 2.18.0 provides this API."
        )
    return quantized_model_init(
        enabled=True,
        recipe=recipe,
        preserve_high_precision_init_val=False,
    )


def _copy_weight_to_primary_fp8(destination, source, recipe) -> None:
    """Calibrate a static E4M3 weight, then quantize it into TE storage."""
    quantizer_getter = getattr(destination, "_get_quantizer", None)
    if quantizer_getter is not None:
        quantizer = quantizer_getter()
        scale = getattr(quantizer, "scale", None)
        scale_inv = getattr(destination, "_scale_inv", None)
        if scale is not None and scale_inv is not None:
            # DelayedScaling would otherwise reuse the random initialization
            # scale for this first copy. Static inference weights can be
            # calibrated exactly once from their checkpoint amax.
            amax = source.detach().abs().amax().float()
            fp8_max = 448.0  # E4M3 finite maximum
            margin = int(getattr(recipe, "margin", 0))
            calibrated_scale = (fp8_max / amax.clamp_min(
                torch.finfo(torch.float32).tiny)) / (2 ** margin)
            scale.copy_(calibrated_scale.to(scale))
            scale_inv.copy_(calibrated_scale.reciprocal().to(scale_inv))

    destination.copy_(source)


class TransformerEngineFP8FFN(nn.Module):
    """FFN with FP8-only primary weights and FP8 E4M3 GEMM execution."""

    uses_fp8 = True

    def __init__(self, source: nn.Sequential, recipe, name: str):
        super().__init__()
        if (
            len(source) != 3
            or not isinstance(source[0], nn.Linear)
            or not isinstance(source[1], nn.GELU)
            or not isinstance(source[2], nn.Linear)
        ):
            raise TypeError("FP8 FFN expects Linear -> GELU -> Linear")

        te, _, _ = _load_transformer_engine()
        first, activation, second = source
        if first.in_features % 16 or first.out_features % 16:
            raise ValueError(
                f"{name}.ffn[0] shape is not FP8 GEMM eligible: "
                f"{first.in_features}x{first.out_features}"
            )
        if second.in_features % 16 or second.out_features % 16:
            raise ValueError(
                f"{name}.ffn[2] shape is not FP8 GEMM eligible: "
                f"{second.in_features}x{second.out_features}"
            )

        device = first.weight.device
        dtype = first.weight.dtype
        # TE only honors quantized_model_init during no-grad construction.
        # params_dtype remains BF16 as the nominal compute/output dtype, while
        # the actual primary weight storage is a TE Float8Tensor.
        with torch.no_grad(), _primary_fp8_init_context(te, recipe):
            self.fc1 = te.Linear(
                first.in_features,
                first.out_features,
                bias=first.bias is not None,
                params_dtype=dtype,
                device=device,
                name=f"{name}.fc1",
            )
            self.fc2 = te.Linear(
                second.in_features,
                second.out_features,
                bias=second.bias is not None,
                params_dtype=dtype,
                device=device,
                name=f"{name}.fc2",
            )

        if not (
            getattr(self.fc1, "primary_weights_in_fp8", False)
            and getattr(self.fc2, "primary_weights_in_fp8", False)
        ):
            raise RuntimeError(
                f"{name}: Transformer Engine did not create FP8 primary "
                "weights. Check that TE 2.18.0 is installed and that "
                "quantized_model_init is enabled."
            )

        # QuantizedTensor.copy_ quantizes the BF16 checkpoint tensor directly
        # into the FP8 destination and updates its scale metadata.
        with torch.no_grad():
            _copy_weight_to_primary_fp8(
                self.fc1.weight, first.weight, recipe)
            _copy_weight_to_primary_fp8(
                self.fc2.weight, second.weight, recipe)
            if first.bias is not None:
                self.fc1.bias.copy_(first.bias)
            if second.bias is not None:
                self.fc2.bias.copy_(second.bias)

        self.activation = nn.GELU(approximate=activation.approximate)
        self.recipe = recipe
        self.primary_weights_in_fp8 = True
        self.forward_calls = 0

    def forward(self, hidden_states):
        # Transformer Engine FP8 GEMM requires:
        #   M = product(hidden_states.shape[:-1]) to be divisible by 8
        #   K = hidden_states.shape[-1] to be divisible by 16
        #
        # Sequence parallelism can produce local token counts such as 387 or
        # 386. FFN is token-wise, so pad only inside the FFN and remove the
        # padding before returning. Padded tokens never enter attention.
        original_shape = hidden_states.shape
        hidden_size = original_shape[-1]
        hidden_states = hidden_states.reshape(-1, hidden_size)
        num_tokens = hidden_states.shape[0]
        padded_tokens = (-num_tokens) % 8
        if padded_tokens:
            hidden_states = F.pad(
                hidden_states, (0, 0, 0, padded_tokens), value=0.0)

        # The weights are already primary Float8Tensors. Do not request the
        # BF16->FP8 weight workspace/cache used by regular TE Linear modules.
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)

        if padded_tokens:
            hidden_states = hidden_states[:num_tokens]
        hidden_states = hidden_states.reshape(*original_shape[:-1], -1)

        self.forward_calls += 1
        return hidden_states


class TransformerEngineFP8Linear(nn.Module):
    """Drop-in inference Linear with FP8-only primary weight storage."""

    uses_fp8 = True
    primary_weights_in_fp8 = True

    def __init__(self, source: nn.Linear, recipe, name: str):
        super().__init__()
        if not isinstance(source, nn.Linear):
            raise TypeError("FP8 Linear conversion expects nn.Linear")
        if source.in_features % 16 or source.out_features % 16:
            raise ValueError(
                f"{name} shape is not FP8 GEMM eligible: "
                f"{source.in_features}x{source.out_features}"
            )

        te, _, _ = _load_transformer_engine()
        with torch.no_grad(), _primary_fp8_init_context(te, recipe):
            self.linear = te.Linear(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
                params_dtype=source.weight.dtype,
                device=source.weight.device,
                name=name,
            )
        if not getattr(self.linear, "primary_weights_in_fp8", False):
            raise RuntimeError(
                f"{name}: Transformer Engine did not create an FP8 primary "
                "weight. Check the TE 2.18.0 installation."
            )

        with torch.no_grad():
            _copy_weight_to_primary_fp8(
                self.linear.weight, source.weight, recipe)
            if source.bias is not None:
                self.linear.bias.copy_(source.bias)

        self.in_features = source.in_features
        self.out_features = source.out_features
        self.forward_calls = 0

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1])
        num_tokens = hidden_states.shape[0]
        padded_tokens = (-num_tokens) % 8
        if padded_tokens:
            hidden_states = F.pad(
                hidden_states, (0, 0, 0, padded_tokens), value=0.0)

        hidden_states = self.linear(hidden_states)

        if padded_tokens:
            hidden_states = hidden_states[:num_tokens]
        hidden_states = hidden_states.reshape(
            *original_shape[:-1], self.out_features)
        self.forward_calls += 1
        return hidden_states


def convert_model_ffn_to_fp8(
    model,
    *,
    start_block: int,
    end_block: int,
) -> dict:
    """Replace inclusive middle-block FFNs with Transformer Engine modules."""
    if start_block < 0 or end_block < start_block:
        raise ValueError("invalid FP8 block range")
    if end_block >= len(model.blocks):
        raise ValueError(
            f"FP8 end block {end_block} exceeds model depth "
            f"{len(model.blocks)}"
        )

    recipe = model.fp8_recipe or make_fp8_recipe()
    converted = []
    for index in range(start_block, end_block + 1):
        block = model.blocks[index]
        if getattr(block.ffn, "uses_fp8", False):
            converted.append(index)
            continue
        block.ffn = TransformerEngineFP8FFN(
            block.ffn, recipe, name=f"blocks.{index}.ffn")
        block.ffn.eval().requires_grad_(False)
        converted.append(index)

    model.fp8_ffn_enabled = True
    model.fp8_enabled = True
    model.fp8_recipe = recipe
    model.fp8_block_range = (start_block, end_block)
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "enabled": True,
        "recipe": "DelayedScaling",
        "format": "E4M3",
        "weight_storage": "primary_fp8",
        "bf16_master_weights": False,
        "reduce_amax": False,
        "blocks": converted,
    }


def convert_model_qkv_to_fp8(
    model,
    *,
    start_block: int,
    end_block: int,
) -> dict:
    """Convert selected fused self-attention QKV projections to FP8."""
    if start_block < 0 or end_block < start_block:
        raise ValueError("invalid FP8 QKV block range")
    if end_block >= len(model.blocks):
        raise ValueError(
            f"FP8 QKV end block {end_block} exceeds model depth "
            f"{len(model.blocks)}"
        )
    if not getattr(model, "self_qkv_fused", False):
        raise ValueError(
            "QKV FP8 requires fused self-attention QKV projections")

    recipe = model.fp8_recipe or make_fp8_recipe()
    converted = []
    for index in range(start_block, end_block + 1):
        attention = model.blocks[index].self_attn
        if attention.qkv is None:
            raise RuntimeError(
                f"blocks.{index}.self_attn has no fused QKV projection")
        if getattr(attention.qkv, "uses_fp8", False):
            converted.append(index)
            continue
        attention.qkv = TransformerEngineFP8Linear(
            attention.qkv,
            recipe,
            name=f"blocks.{index}.self_attn.qkv",
        )
        attention.qkv.eval().requires_grad_(False)
        converted.append(index)

    model.fp8_qkv_enabled = True
    model.fp8_enabled = True
    model.fp8_recipe = recipe
    model.fp8_qkv_block_range = (start_block, end_block)
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "enabled": True,
        "recipe": "DelayedScaling",
        "format": "E4M3",
        "weight_storage": "primary_fp8",
        "bf16_master_weights": False,
        "reduce_amax": False,
        "blocks": converted,
    }


def fp8_model_stats(model) -> dict:
    ffn_blocks = []
    ffn_calls = 0
    qkv_blocks = []
    qkv_calls = 0
    for index, block in enumerate(model.blocks):
        if getattr(block.ffn, "uses_fp8", False):
            ffn_blocks.append(index)
            ffn_calls += int(block.ffn.forward_calls)
        if getattr(block.self_attn.qkv, "uses_fp8", False):
            qkv_blocks.append(index)
            qkv_calls += int(block.self_attn.qkv.forward_calls)
    enabled = bool(ffn_blocks or qkv_blocks)
    return {
        "enabled": enabled,
        "ffn": {
            "enabled": bool(ffn_blocks),
            "blocks": ffn_blocks,
            "forward_calls": ffn_calls,
        },
        "qkv": {
            "enabled": bool(qkv_blocks),
            "blocks": qkv_blocks,
            "forward_calls": qkv_calls,
        },
        "primary_weights_in_fp8": all(
            model.blocks[index].ffn.primary_weights_in_fp8
            for index in ffn_blocks
        ) and all(
            model.blocks[index].self_attn.qkv.primary_weights_in_fp8
            for index in qkv_blocks
        ) if enabled else False,
        "bf16_master_weights": False if enabled else None,
    }
