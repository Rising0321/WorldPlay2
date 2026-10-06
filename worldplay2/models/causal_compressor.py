# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Causal memory compressor for WorldPlay2.

This is the Wan2.2 inference-side port of the FastVideo training module
``fastvideo/models/hyvideo/models/transformers/modules/causal_memory_compress.py``
(``CausalMemCompressModel`` / ``CausalCompressConv3D``).

The ONLY difference from the non-causal ``MemCompressModel`` is that every
convolution spanning the TEMPORAL axis is made causal: an output frame at time
``t`` may depend on input frames ``<= t`` (current + past) only, never on
future frames.  This is achieved by padding the temporal axis on the LEFT
(past) side only -- exactly like ``CausalConv3d`` in the Wan VAE
(``worldplay2/vae/wan21.py``) and ``WanCausalConv3d`` in the FastVideo wanvae.

Concretely (matching the FastVideo training module verbatim):
  - ``self.conv`` (kernel 3, temporal padding 1) becomes a ``CausalConv3d``:
    instead of symmetric temporal padding (1 left + 1 right) it pads
    ``2*1 = 2`` frames on the left and 0 on the right, so the output length is
    unchanged but each frame only sees the past.
  - ``self.time_conv`` (kernel 3, temporal stride 2 -- the temporal
    downsample) becomes a ``CausalConv3d`` with temporal padding 1 -> 2 frames
    of left padding, 0 right.  The old non-causal implementation zero-padded 1
    frame on the RIGHT (future) side, which leaked one future frame into every
    downsampled output.

    OUTPUT-LENGTH INVARIANT: the temporal downsample must still map
    ``T_hr -> T_hr/2`` so ``hr_tokens`` line up token-by-token with
    ``lr_tokens`` (which live at ``T_lr = T_hr/2``).  For the wan latent
    cadence ``T_hr`` is always EVEN; with an even input:
      old (right-pad 1):  out = (T_hr + 1 - 3)//2 + 1 = T_hr/2
      new (left-pad 2):   out = (T_hr + 2 - 3)//2 + 1 = T_hr/2
    so the output length is IDENTICAL -- only the temporal receptive field
    shifts from centered to left/causal.
  - ``self.resample`` only touches H/W (spatial), so it is left unchanged.
  - ``_AvgDown3D`` (the temporal shortcut) already pads temporal on the left
    only, so it is already causal.

The compressor here is run over the WHOLE history in one shot (not streamed),
so ``CausalConv3d`` is always called with ``cache_x=None`` (standard causal
left-pad).  The ``cache_x`` streaming interface is inherited but unused.
"""

import torch
import torch.nn as nn
from einops import rearrange

# Reuse the causal 3d conv from the Wan VAE (identical padding semantics to
# FastVideo's WanCausalConv3d).
from ..vae.wan21 import CausalConv3d
# Reuse the (already-causal / time-independent) helper blocks from the
# non-causal compressor so weights and shapes match the training module.
from .compressor_blocks import _AvgDown3D, _WanAttentionBlock


class CausalCompressConv3D(nn.Module):
    """Causal-in-time version of ``memory_compress.CompressConv3D``.

    Same structure and parameter layout as ``CompressConv3D`` except the two
    temporal-spanning convolutions (``conv`` and ``time_conv``) are causal.
    """

    def __init__(self, in_dim, out_dim, spatial_down, temporal_down,
                 non_linearity='silu'):
        super().__init__()
        # Causal in time: kernel 3 with padding 1 -> left-only temporal pad.
        self.conv = CausalConv3d(in_channels=in_dim, out_channels=out_dim,
                                 kernel_size=3, stride=1, padding=1)
        if non_linearity == 'silu':
            self.nonlinearity = nn.SiLU()
        else:
            raise ValueError(
                f"Unsupported non_linearity={non_linearity!r}. "
                "Only 'silu' is used by Wan2.2 5B; extend if needed.")
        self.spatial_down = spatial_down
        self.temporal_down = temporal_down

        if self.spatial_down:
            # Spatial-only downsample -- no temporal axis touched, unchanged.
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(out_dim, out_dim, 3, stride=(2, 2)),
            )

        if self.temporal_down:
            # Causal temporal downsample: kernel 3, stride 2, temporal
            # padding 1 -> CausalConv3d pads 2 frames on the LEFT (past) and 0
            # on the right, so no future frame leaks into the output.
            self.time_conv = CausalConv3d(
                out_dim, out_dim, kernel_size=(3, 1, 1),
                stride=(2, 1, 1), padding=(1, 0, 0))

        if self.spatial_down or self.temporal_down:
            self.avg_shortcut = _AvgDown3D(
                in_dim, out_dim,
                factor_t=2 if temporal_down else 1,
                factor_s=2 if spatial_down else 1,
            )
        else:
            self.avg_shortcut = nn.Conv3d(in_dim, out_dim, kernel_size=1,
                                          stride=1, padding=0)

    def forward(self, x):
        x_copy = x.clone()
        x = self.conv(x)
        x = self.nonlinearity(x)
        b, c, t, h, w = x.shape
        if self.spatial_down:
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = self.resample(x)
            x = x.view(b, t, x.size(1), x.size(2),
                       x.size(3)).permute(0, 2, 1, 3, 4)
        if self.temporal_down:
            x = self.time_conv(x)
        return x + self.avg_shortcut(x_copy)


class CausalMemCompressModel(nn.Module):
    """Causal-in-time variant of ``memory_compress.MemCompressModel``.

    Identical to ``MemCompressModel`` except each ``CompressConv3D`` block is
    replaced by ``CausalCompressConv3D`` so the temporal receptive field is
    strictly causal (an output token never depends on future latent frames).
    The ``attn_blocks`` are per-frame spatial attention (no temporal mixing),
    so they are already causal and reused verbatim.
    """

    def __init__(
        self,
        output_dim=2048,
        input_dim=65,
        dims=(64, 64, 128, 256, 256, 512, 512),
        spatial_down=(1, 1, 0, 0, 0, 0),
        temporal_down=(1, 0, 0, 0, 0, 0),
        attn_num=2,
        non_linearity: str = "silu",
    ):
        super().__init__()
        dims = list(dims)
        spatial_down = list(spatial_down)
        temporal_down = list(temporal_down)

        self.input_layer = nn.Linear(input_dim, dims[0])

        self.blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self.blocks.append(CausalCompressConv3D(
                in_dim, out_dim, spatial_down[i], temporal_down[i],
                non_linearity=non_linearity))

        self.attn_blocks = nn.ModuleList(
            [_WanAttentionBlock(dims[-1], head_dim=8) for _ in range(attn_num)])

        self.output_layer = nn.Linear(dims[-1], output_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'B C T H W -> B (T H W) C')
        x = self.input_layer(x)
        x = rearrange(x, 'B (T H W) C -> B C T H W', T=T, H=H, W=W)
        for block in self.blocks:
            x = block(x)
        for attn_block in self.attn_blocks:
            x = attn_block(x)
        x = rearrange(x, 'B C T H W -> B (T H W) C')
        return self.output_layer(x)
