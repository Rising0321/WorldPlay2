# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Hierarchical HR -> compressed-memory feature extractor used by the
stage-2 / stage-3 chunked inference recipe.

This is a self-contained port of ``MemCompressModel`` from the FastVideo
training pipeline.  It only depends on ``torch.nn`` plus ``einops`` so
that it can be imported from ``worldplay2/models/model.py`` without dragging in
any FastVideo-specific layers / autocast helpers.

Wan2.2 5B contract (matches ``WorldPlay2Model.add_memory_compress_model``):
    input_dim     = 36   (16 hr-latent + 20 hr-y)
    spatial_down  = [1, 1, 1, 0, 0, 0]   (3 spatial down blocks * 2 each = /8)
    temporal_down = [1, 0, 0, 0, 0, 0]   (1 temporal down block * 2 = /2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Local copies of the 3 helpers MemCompressModel uses from FastVideo.
# Kept verbatim so checkpoints trained against FastVideo load directly.
# ---------------------------------------------------------------------------

class _AvgDown3D(nn.Module):
    """Channel-folding average-pool used as the residual shortcut inside
    ``CompressConv3D``.  Identical to ``fastvideo.models.vaes.wanvae.AvgDown3D``.
    """

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B, C,
            T // self.factor_t, self.factor_t,
            H // self.factor_s, self.factor_s,
            W // self.factor_s, self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B, C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B, self.out_channels, self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        return x.mean(dim=2)


class _WanRMS_norm(nn.Module):
    """Same as ``fastvideo.models.vaes.wanvae.WanRMS_norm``.  Operates
    per-frame on (B*T, C, H, W) tensors.
    """

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)
        ) * self.scale * self.gamma + self.bias


class _WanAttentionBlock(nn.Module):
    """Multi-head per-frame self-attention.  Identical to
    ``fastvideo.models.vaes.wanvae.WanAttentionBlock``; this is the
    ``head_dim=8`` variant used by ``MemCompressModel``.
    """

    def __init__(self, dim, head_dim=1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.norm = _WanRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b h l (H d) -> b (h H) l d", H=self.head_dim)
        k = rearrange(k, "b h l (H d) -> b (h H) l d", H=self.head_dim)
        v = rearrange(v, "b h l (H d) -> b (h H) l d", H=self.head_dim)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l d -> b (h d) l")

        x = x.reshape(batch_size * time, channels, height, width)
        x = self.proj(x)

        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        return x + identity

