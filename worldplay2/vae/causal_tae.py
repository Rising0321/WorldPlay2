#!/usr/bin/env python3
"""
Causal (left-padded) TAE-style ENCODER for Wan 2.1, adapted for 4n+1 frame
video inputs.

This is a variant of the TAEW2_1 encoder (see ``taew2_1.py``) with ONE
architectural change in the temporal-pooling behaviour:

    * The original ``TPool`` reshapes every ``stride`` consecutive frames
      into the channel axis directly. It relied on ``encode_video`` padding
      the time axis at the *end* to a multiple of ``t_downscale``.

    * ``CausalTPool`` (here) instead pads ONE zero frame on the *left*
      (the temporal-causal side) whenever the incoming time length is not a
      multiple of ``stride`` (i.e. for stride>1 on an odd-length input),
      then pools. For a 4n+1 input this cascades cleanly:
          T=4n+1 --pad1--> 4n+2 --pool2--> 2n+1 --pad1--> 2n+2 --pool2--> n+1.

``MemBlock`` keeps the SAME behaviour as the original parallel path: it
takes the "previous frame" as memory by left-padding one zero frame along
the time axis and concatenating it on the channel dim inside the block.
That part is unchanged — this file only re-implements the pooling.

Scope (per request):
    * ENCODER ONLY. No decoder / TGrow.
    * ONE-SHOT (parallel) encode only. No streaming work-queue path.

I/O convention (same as TAEW2_1):
    encode_video: input  [N, T, C=3, H, W] in [0, 1], T = 4n+1
                  output [N, T_lat, latent_channels, H/8, W/8]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class MemBlock(nn.Module):
    """Residual block that also consumes the previous frame's activation
    (its "memory") concatenated on the channel axis. Identical to the
    TAEW2_1 MemBlock."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

    def forward_chunk(self, x, N, cache):
        """Streaming forward over one chunk in [N*T, C, H, W] layout.

        The per-frame memory is the *previous* frame's input. Within the
        chunk this is the left-shifted version of ``x``; the very first
        frame of the chunk uses ``cache`` (the last input frame of the
        PREVIOUS chunk) instead of a zero frame -- except when ``cache`` is
        None (the first chunk overall), where it falls back to zeros. This
        makes the chunked result identical to the one-shot path.

        Args:
            x: [N*T, C, H, W] chunk input.
            N: batch size.
            cache: previous chunk's last-frame input, [N, 1, C, H, W], or None.
        Returns:
            (out, new_cache) where out is [N*T, C, H, W] and new_cache is
            this chunk's last-frame input [N, 1, C, H, W].
        """
        NT, C, H, W = x.shape
        T = NT // N
        _x = x.view(N, T, C, H, W)
        if cache is None:
            head = _x.new_zeros(N, 1, C, H, W)     # first chunk: zero history
        else:
            head = cache                            # continuation: prev last frame
        # memory[t] = input[t-1]; memory[0] = head
        block_memory = torch.cat([head, _x[:, :T - 1]], dim=1).reshape(x.shape)
        new_cache = _x[:, -1:].clone()
        out = self.forward(x, block_memory)
        return out, new_cache


class CausalTPool(nn.Module):
    """Temporal pooling that LEFT-pads one zero frame when needed so a
    4n+1 (odd) time length becomes even before pooling every ``stride``
    frames into the channel axis.

    Operates on the flattened [N*T, C, H, W] layout used by the parallel
    encoder path, and therefore needs ``N`` (batch size) to recover the
    time dimension.
    """

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x, N):
        # x: [N*T, C, H, W]
        NT, C, H, W = x.shape
        T = NT // N
        if self.stride == 1:
            # no temporal pooling; the 1x1 conv still maps n_f -> n_f
            return self.conv(x)

        x = x.view(N, T, C, H, W)
        # left-pad zeros along the time axis until T is a multiple of stride
        # (for 4n+1 inputs and stride=2 this pads exactly one frame).
        rem = T % self.stride
        if rem != 0:
            n_pad = self.stride - rem
            pad = x.new_zeros(N, n_pad, C, H, W)
            x = torch.cat([pad, x], dim=1)  # LEFT pad (causal side)
            T = x.shape[1]

        # group every ``stride`` consecutive frames onto the channel axis,
        # matching the original TPool's reshape ordering (frame t occupies
        # channels [0:C], frame t+1 -> [C:2C], ...).
        x = x.view(N, T // self.stride, self.stride * C, H, W)
        x = x.reshape(N * (T // self.stride), self.stride * C, H, W)
        return self.conv(x)

    def forward_chunk(self, x, N, is_first_chunk):
        """Streaming pool for the 1,4,4,4 chunk schedule.

        With that schedule and stride=2, only the FIRST chunk is odd
        (length 1 -> left-pad 1 -> 2 -> pool -> 1); every subsequent chunk
        already has a length that is a multiple of ``stride`` (4 -> pool ->
        2, and at the 2nd pooling stage 2 -> pool -> 1), so no cross-chunk
        residual buffering is needed and the concatenation of per-chunk
        outputs is bit-for-bit identical to the one-shot ``forward``.

        Args:
            x: [N*T, C, H, W] chunk input.
            N: batch size.
            is_first_chunk: True only for the very first chunk.
        Returns:
            [N*(T_out), C, H, W] pooled output for this chunk.
        """
        NT, C, H, W = x.shape
        T = NT // N
        if self.stride == 1:
            return self.conv(x)

        _x = x.view(N, T, C, H, W)
        rem = T % self.stride
        if rem != 0:
            assert is_first_chunk, (
                f"CausalTPool.forward_chunk: non-first chunk has time length "
                f"{T} which is not a multiple of stride {self.stride}; the "
                f"1,4,4,4 schedule should keep continuation chunks aligned.")
            n_pad = self.stride - rem
            pad = _x.new_zeros(N, n_pad, C, H, W)
            _x = torch.cat([pad, _x], dim=1)  # LEFT pad only on the first chunk
            T = _x.shape[1]

        _x = _x.view(N, T // self.stride, self.stride * C, H, W)
        _x = _x.reshape(N * (T // self.stride), self.stride * C, H, W)
        return self.conv(_x)


class Clamp(nn.Module):
    """Soft clamp to ~[-3, 3] on the latent, used at the head of the decoder
    (identical to the TAEW2_1 Clamp)."""

    def forward(self, x):
        return torch.tanh(x / 3) * 3


class CausalTGrow(nn.Module):
    """Temporal upsampling, inverse of ``CausalTPool``.

    ``conv`` expands the channel axis to ``stride * n_f`` and the frames are
    then split back out along the time axis (1 latent frame -> ``stride``
    pixel-time frames), matching the original TGrow reshape ordering.

    For CAUSAL first-frame handling, the VERY FIRST latent frame of a clip
    maps to a single output frame with ``output == input`` (it skips both the
    conv and the temporal expansion entirely), while every subsequent latent
    frame is expanded to ``stride`` frames. This mirrors the encoder, where
    the first pixel frame maps to one latent; here the first latent maps back
    to one frame, so:

        T_lat = n + 1  ->  T_pix = 1 + n * (product of TGrow strides) = 4n+1
    """

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def _expand(self, x):
        """conv + reshape: [N*T, C, H, W] -> [N*T*stride, C, H, W]."""
        NT, C, H, W = x.shape
        x = self.conv(x)                       # [N*T, stride*C, H, W]
        return x.reshape(NT * self.stride, C, H, W)

    def forward(self, x, N, is_first_clip=True):
        """One-shot temporal grow over the whole clip.

        Args:
            x: [N*T_lat, C, H, W].
            N: batch size.
            is_first_clip: if True, the first latent frame of the clip
                expands to a single output frame (causal startup), the rest
                expand to ``stride`` frames each. If False (continuation
                chunk in streaming decode), ALL frames expand to ``stride``.
        Returns:
            [N*T_out, C, H, W].
        """
        NT, C, H, W = x.shape
        T = NT // N
        if self.stride == 1:
            # no temporal upsampling; conv maps n_f -> n_f*1 = n_f, reshape no-op
            return self._expand(x)

        _x = x.view(N, T, C, H, W)
        if is_first_clip:
            # first latent frame: output == input (no temporal expansion,
            # no conv), so it maps to exactly ONE output frame.
            first = _x[:, :1]                             # [N, 1, C, H, W]
            rest = _x[:, 1:].reshape(N * (T - 1), C, H, W) if T > 1 else None
            outs = [first]
            if rest is not None:
                rest_grown = self._expand(rest)           # [N*(T-1)*stride, C, H, W]
                rest_grown = rest_grown.view(N, (T - 1) * self.stride, C, H, W)
                outs.append(rest_grown)
            out = torch.cat(outs, dim=1)                  # [N, T_out, C, H, W]
            return out.reshape(N * out.shape[1], C, H, W)
        else:
            # continuation chunk: every latent frame expands to stride frames
            return self._expand(x)


class CausalTAEEncoder(nn.Module):
    """One-shot causal TAE-style encoder for Wan 2.1 (4n+1 frame inputs)."""

    def __init__(self,
                 encoder_time_downscale=(True, True, False),
                 latent_channels=16,
                 patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size ** 2, 64), nn.ReLU(inplace=True),
            CausalTPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            CausalTPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            CausalTPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )

        # temporal downscale factor (product of the pooling strides)
        self.t_downscale = 1
        for m in self.encoder:
            if isinstance(m, CausalTPool):
                self.t_downscale *= m.stride

    def preprocess_input_frames(self, x):
        """Preprocess RGB input frames prior to the main encoder sequence."""
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        return x

    def _apply_encoder(self, x, show_progress_bar):
        """Parallel (one-shot) application of the encoder sequence over the
        time axis.

        Args:
            x: [N, T, C, H, W] tensor.
        Returns:
            [N, T_lat, latent_channels, H/8, W/8] tensor.
        """
        assert x.ndim == 5, f"expected NTCHW tensor, got {x.ndim}-dim"
        N, T, C, H, W = x.shape
        x = x.reshape(N * T, C, H, W)

        for b in tqdm(self.encoder, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                # "previous frame" memory = left-pad one zero frame along
                # time, then drop the last frame so lengths line up. This is
                # the same causal memory the original parallel path uses.
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                block_memory = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = b(x, block_memory)
            elif isinstance(b, CausalTPool):
                # CausalTPool needs N to recover the time dim and does the
                # left-pad-then-pool internally.
                x = b(x, N)
            else:
                x = b(x)

        NT, C, H, W = x.shape
        T = NT // N
        return x.view(N, T, C, H, W)

    def encode_video(self, x, show_progress_bar=True):
        """Encode a sequence of frames (one-shot / parallel).

        Args:
            x: input [N, T, C=3, H, W] RGB tensor in [0, 1], with T = 4n+1.
            show_progress_bar: enable tqdm over encoder blocks.
        Returns:
            [N, T_lat, latent_channels, H/8, W/8] latent tensor.

        Unlike TAEW2_1.encode_video, there is NO end-of-sequence padding:
        the left-padding inside each CausalTPool handles odd (4n+1) lengths
        causally.
        """
        x = self.preprocess_input_frames(x)
        return self._apply_encoder(x, show_progress_bar)

    # ------------------------------------------------------------------
    # streaming / chunked encode (1, 4, 4, 4, ... schedule)
    # ------------------------------------------------------------------

    def init_cache(self):
        """Create a fresh per-block streaming cache.

        Returns a list with one slot per encoder block; only ``MemBlock``
        slots are ever populated (each holds that block's last-frame input
        from the previous chunk). Pass the same object to every
        :meth:`encode_chunk` call of one stream.
        """
        return [None] * len(self.encoder)

    def _apply_encoder_chunk(self, x, cache, is_first_chunk):
        """Streaming application of the encoder over one chunk.

        Args:
            x: [N, T_chunk, C, H, W] chunk (T_chunk == 1 for the first chunk,
               then 4 for each continuation chunk).
            cache: list from :meth:`init_cache`, mutated in place.
            is_first_chunk: True only for the first chunk of the stream.
        Returns:
            [N, T_lat_chunk, latent_channels, H/8, W/8] latent for this chunk.
        """
        assert x.ndim == 5, f"expected NTCHW tensor, got {x.ndim}-dim"
        N = x.shape[0]
        x = x.reshape(N * x.shape[1], *x.shape[2:])

        for i, b in enumerate(self.encoder):
            if isinstance(b, MemBlock):
                x, cache[i] = b.forward_chunk(x, N, cache[i])
            elif isinstance(b, CausalTPool):
                x = b.forward_chunk(x, N, is_first_chunk)
            else:
                x = b(x)

        NT, C, H, W = x.shape
        return x.view(N, NT // N, C, H, W)

    def encode_chunk(self, x, cache, is_first_chunk=False):
        """Encode ONE chunk of frames, continuing the streaming state.

        Feed the video in the ``1, 4, 4, 4, ...`` schedule: the first call
        with ``is_first_chunk=True`` passes a single frame ([N, 1, C, H, W]),
        and each subsequent call passes 4 frames ([N, 4, C, H, W]) with
        ``is_first_chunk=False``. Concatenating the returned latents along
        the time axis reproduces :meth:`encode_video` on the full clip
        exactly.

        Args:
            x: [N, T_chunk, C=3, H, W] RGB chunk in [0, 1].
            cache: list from :meth:`init_cache`, reused across the stream.
            is_first_chunk: True only for the very first chunk.
        Returns:
            [N, T_lat_chunk, latent_channels, H/8, W/8] latent for this chunk.
        """
        x = self.preprocess_input_frames(x)
        return self._apply_encoder_chunk(x, cache, is_first_chunk)

    def encode_video_chunked(self, x, show_progress_bar=False):
        """Convenience wrapper: split a full 4n+1 clip into the 1,4,4,4,...
        schedule, run :meth:`encode_chunk` over each piece, and concatenate.

        Args:
            x: [N, T=4n+1, C=3, H, W] RGB clip in [0, 1].
        Returns:
            [N, T_lat, latent_channels, H/8, W/8] latent (== encode_video).
        """
        T = x.shape[1]
        assert T % 4 == 1, f"chunked encode expects 4n+1 frames, got T={T}"
        cache = self.init_cache()
        outs = []
        # first chunk: 1 frame
        outs.append(self.encode_chunk(x[:, :1], cache, is_first_chunk=True))
        # continuation chunks: 4 frames each
        starts = range(1, T, 4)
        for s in tqdm(starts, disable=not show_progress_bar):
            outs.append(self.encode_chunk(x[:, s:s + 4], cache, is_first_chunk=False))
        return torch.cat(outs, dim=1)

    def forward(self, x, show_progress_bar=False):
        return self.encode_video(x, show_progress_bar=show_progress_bar)


class CausalTAEDecoder(nn.Module):
    """Causal TAE-style decoder for Wan 2.1, inverse of CausalTAEEncoder.

    The architecture mirrors the TAEW2_1 base decoder. The only causal
    change is in temporal upsampling: the FIRST latent frame of a clip
    decodes to a SINGLE output frame (via CausalTGrow's first-frame path),
    so the whole decoder maps

        T_lat = n + 1  latent frames  ->  T_pix = 4n + 1  output frames

    which is the exact inverse of the encoder. No end-of-sequence trimming
    is required (unlike the original TAEW2_1.decode_video which trims
    ``frames_to_trim`` startup frames).
    """

    def __init__(self,
                 decoder_time_upscale=(False, True, True),
                 decoder_space_upscale=(True, True, True),
                 latent_channels=16,
                 patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3

        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), CausalTGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), CausalTGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), CausalTGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], self.image_channels * self.patch_size ** 2),
        )

        # temporal upscale factor (product of the grow strides)
        self.t_upscale = 1
        for m in self.decoder:
            if isinstance(m, CausalTGrow):
                self.t_upscale *= m.stride

    def postprocess_output_frames(self, x):
        """Postprocess RGB frames after the main decoder sequence."""
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x.clamp_(0, 1)

    def _apply_decoder(self, x, is_first_clip, show_progress_bar):
        """Apply the decoder sequence over the time axis (one-shot).

        Args:
            x: [N, T_lat, C, H, W] latent tensor.
            is_first_clip: True if this batch of latents starts a clip (the
                first latent frame decodes to a single output frame).
        Returns:
            [N, T_out, image_channels*patch_size**2, H*8, W*8] tensor.
        """
        assert x.ndim == 5, f"expected NTCHW tensor, got {x.ndim}-dim"
        N, T, C, H, W = x.shape
        x = x.reshape(N * T, C, H, W)

        for b in tqdm(self.decoder, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                block_memory = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = b(x, block_memory)
            elif isinstance(b, CausalTGrow):
                x = b(x, N, is_first_clip=is_first_clip)
            else:
                x = b(x)

        NT, C, H, W = x.shape
        T = NT // N
        return x.view(N, T, C, H, W)

    def decode_video(self, x, show_progress_bar=True):
        """Decode a full sequence of latents (one-shot / parallel).

        Args:
            x: input [N, T_lat=n+1, C=latent_channels, H, W] latent tensor.
            show_progress_bar: enable tqdm over decoder blocks.
        Returns:
            [N, T_pix=4n+1, 3, H*8, W*8] RGB tensor in [0, 1].
        """
        x = self._apply_decoder(x, is_first_clip=True, show_progress_bar=show_progress_bar)
        return self.postprocess_output_frames(x)

    def decode_latent(self, x, is_first_latent, show_progress_bar=False):
        """Decode ONE (or a few) latent frame(s) with explicit first-frame
        control, WITHOUT threading a cross-latent cache -- the low-level
        stateless building block.

        NOTE: without a MemBlock cache, calling this repeatedly is NOT
        numerically identical to ``decode_video`` at chunk boundaries (each
        call restarts the MemBlock memory from zeros). For an equivalent
        one-latent-at-a-time decode, use :meth:`init_cache` +
        :meth:`decode_chunk` instead.

        Args:
            x: [N, T_sub, C=latent_channels, H, W] latent(s).
            is_first_latent: True if these latents include the clip's first
                latent frame (-> single output frame for that frame).
        Returns:
            [N, T_out, 3, H*8, W*8] RGB tensor in [0, 1].
        """
        x = self._apply_decoder(x, is_first_clip=is_first_latent,
                                show_progress_bar=show_progress_bar)
        return self.postprocess_output_frames(x)

    # ------------------------------------------------------------------
    # streaming / chunked decode (cross-latent cache)
    # ------------------------------------------------------------------

    def init_cache(self):
        """Create a fresh per-block streaming cache.

        Returns a list with one slot per decoder block; only ``MemBlock``
        slots are ever populated (each holds that block's last-frame input
        from the previous chunk). Pass the same object to every
        :meth:`decode_chunk` call of one stream.
        """
        return [None] * len(self.decoder)

    def _apply_decoder_chunk(self, x, cache, is_first_chunk):
        """Streaming application of the decoder over one chunk of latents.

        MemBlock threads its "previous frame" memory through ``cache`` (same
        contract as the encoder), and CausalTGrow uses ``is_first_chunk`` so
        that only the very first latent of the stream yields a single frame
        (output == input); continuation chunks expand every latent frame by
        ``stride``.

        Args:
            x: [N, T_sub, C, H, W] latent chunk.
            cache: list from :meth:`init_cache`, mutated in place.
            is_first_chunk: True only for the first chunk of the stream.
        Returns:
            [N, T_out, image_channels*patch_size**2, H*8, W*8] tensor.
        """
        assert x.ndim == 5, f"expected NTCHW tensor, got {x.ndim}-dim"
        N = x.shape[0]
        x = x.reshape(N * x.shape[1], *x.shape[2:])

        for i, b in enumerate(self.decoder):
            if isinstance(b, MemBlock):
                x, cache[i] = b.forward_chunk(x, N, cache[i])
            elif isinstance(b, CausalTGrow):
                x = b(x, N, is_first_clip=is_first_chunk)
            else:
                x = b(x)

        NT, C, H, W = x.shape
        return x.view(N, NT // N, C, H, W)

    def decode_chunk(self, x, cache, is_first_chunk=False):
        """Decode ONE chunk of latents, continuing the streaming state.

        Feed latents one (or a few) at a time. The first call with
        ``is_first_chunk=True`` decodes the clip's first latent to a single
        output frame; every subsequent call (``is_first_chunk=False``)
        expands each latent frame to ``t_upscale`` output frames.
        Concatenating the returned frames along the time axis reproduces
        :meth:`decode_video` on the full latent sequence exactly.

        Args:
            x: [N, T_sub, C=latent_channels, H, W] latent chunk.
            cache: list from :meth:`init_cache`, reused across the stream.
            is_first_chunk: True only for the very first chunk.
        Returns:
            [N, T_out, 3, H*8, W*8] RGB frames in [0, 1].
        """
        x = self._apply_decoder_chunk(x, cache, is_first_chunk)
        return self.postprocess_output_frames(x)

    def decode_video_streamed(self, x, show_progress_bar=False):
        """Convenience wrapper: decode a full latent sequence one latent at a
        time through :meth:`decode_chunk` and concatenate.

        Numerically identical to :meth:`decode_video` (up to float rounding).

        Args:
            x: [N, T_lat=n+1, C=latent_channels, H, W] latent tensor.
        Returns:
            [N, T_pix=4n+1, 3, H*8, W*8] RGB tensor in [0, 1].
        """
        T = x.shape[1]
        cache = self.init_cache()
        outs = []
        for i in tqdm(range(T), disable=not show_progress_bar):
            outs.append(self.decode_chunk(x[:, i:i + 1], cache,
                                          is_first_chunk=(i == 0)))
        return torch.cat(outs, dim=1)

    def forward(self, x, show_progress_bar=False):
        return self.decode_video(x, show_progress_bar=show_progress_bar)


class CausalTAE(nn.Module):
    """Unified causal TAE for Wan 2.1: encoder + decoder in one module.

    Wraps :class:`CausalTAEEncoder` and :class:`CausalTAEDecoder` so a single
    object exposes the full video<->latent roundtrip. Both one-shot and
    streaming (chunked) paths are provided and are numerically equivalent.

    I/O convention:
        encode: [N, T=4n+1, C=3, H, W] RGB in [0, 1]
                -> [N, T_lat=n+1, latent_channels, H/8, W/8]
        decode: [N, T_lat=n+1, latent_channels, H, W]
                -> [N, T_pix=4n+1, 3, H*8, W*8] RGB in [0, 1]

    Streaming schedules (equivalent to the one-shot paths):
        encode: feed frames as 1, 4, 4, 4, ...   (encode_chunk)
        decode: feed latents one at a time       (decode_chunk)
    """

    def __init__(self,
                 encoder_time_downscale=(True, True, False),
                 decoder_time_upscale=(False, True, True),
                 decoder_space_upscale=(True, True, True),
                 latent_channels=16,
                 patch_size=1):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.image_channels = 3

        self.enc = CausalTAEEncoder(
            encoder_time_downscale=encoder_time_downscale,
            latent_channels=latent_channels,
            patch_size=patch_size,
        )
        self.dec = CausalTAEDecoder(
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
            latent_channels=latent_channels,
            patch_size=patch_size,
        )
        self.t_downscale = self.enc.t_downscale
        self.t_upscale = self.dec.t_upscale

    # ---- one-shot ----------------------------------------------------

    def encode_video(self, x, show_progress_bar=True):
        """One-shot encode. [N, 4n+1, 3, H, W] -> [N, n+1, C_z, H/8, W/8]."""
        return self.enc.encode_video(x, show_progress_bar=show_progress_bar)

    def decode_video(self, z, show_progress_bar=True):
        """One-shot decode. [N, n+1, C_z, H, W] -> [N, 4n+1, 3, H*8, W*8]."""
        return self.dec.decode_video(z, show_progress_bar=show_progress_bar)

    def forward(self, x, teacher_latent=None, show_progress_bar=False):
        """Two modes:

        * ``teacher_latent is None`` (default): full roundtrip encode->decode,
          returns reconstructed RGB.
        * ``teacher_latent`` given: TRAINING mode. Runs both the encoder
          (supervised by ``teacher_latent``) and the decoder (fed the teacher
          latent, reconstructing ``x``) through this single ``forward`` so that
          DistributedDataParallel sees the whole graph and syncs gradients
          correctly. Returns ``(encoded, decoded)``:
              encoded: [N, T_lat, C_z, H/8, W/8]   (compare to teacher_latent)
              decoded: [N, T_pix, 3, H*8, W*8]     (compare to x)
        """
        if teacher_latent is None:
            z = self.encode_video(x, show_progress_bar=show_progress_bar)
            return self.decode_video(z, show_progress_bar=show_progress_bar)
        encoded = self.encode_video(x, show_progress_bar=show_progress_bar)
        decoded = self.decode_video(teacher_latent, show_progress_bar=show_progress_bar)
        return encoded, decoded

    # ---- streaming: encode (1, 4, 4, 4, ...) -------------------------

    def init_encode_cache(self):
        return self.enc.init_cache()

    def encode_chunk(self, x, cache, is_first_chunk=False):
        """Streaming encode of one chunk (1 frame first, then 4 each)."""
        return self.enc.encode_chunk(x, cache, is_first_chunk=is_first_chunk)

    def encode_video_chunked(self, x, show_progress_bar=False):
        """Streaming encode over a full 4n+1 clip (== encode_video)."""
        return self.enc.encode_video_chunked(x, show_progress_bar=show_progress_bar)

    # ---- streaming: decode (one latent at a time) --------------------

    def init_decode_cache(self):
        return self.dec.init_cache()

    def decode_chunk(self, z, cache, is_first_chunk=False):
        """Streaming decode of one latent chunk (first -> 1 frame, then
        t_upscale frames each)."""
        return self.dec.decode_chunk(z, cache, is_first_chunk=is_first_chunk)

    def decode_video_streamed(self, z, show_progress_bar=False):
        """Streaming decode over a full latent sequence (== decode_video)."""
        return self.dec.decode_video_streamed(z, show_progress_bar=show_progress_bar)


# ===========================================================================
# Drop-in adapter exposing the SAME interface as ``worldplay2.vae.wan21.Wan2_1_VAE``
# so ``image2video_stage_two`` can switch VAEs without touching call sites.
# ===========================================================================

class CausalTAEVAE:
    r"""Adapter that wraps :class:`CausalTAE` behind the exact public surface
    of :class:`worldplay2.vae.wan21.Wan2_1_VAE`, so the stage-2/3 pipeline can
    use it as a drop-in replacement with NO changes to its call sites.

    Interface parity (all list-in / list-out, elements are ``[C, T, H, W]``
    tensors WITHOUT a batch dim, pixels in ``[-1, 1]``):
        * ``encode([clip])``               -> ``[z]``
        * ``decode([z])``                  -> ``[clip]``
        * ``encode_chunk([clip], is_first_chunk)`` -> ``[z]``
        * ``decode_chunk([z],   is_first_chunk)``  -> ``[clip]``
      plus the ``.scale`` / ``.dtype`` / ``.device`` attributes.

    Differences from ``CausalTAE`` that this adapter bridges:
      * LAYOUT: ``Wan2_1_VAE`` uses list of ``[C, T, H, W]``; ``CausalTAE``
        uses batched ``[N, T, C, H, W]``.  We add/remove the batch dim and
        move the channel axis.
      * VALUE RANGE: ``Wan2_1_VAE`` pixels are in ``[-1, 1]``; ``CausalTAE``
        pixels are in ``[0, 1]``.  We map ``(x+1)/2`` in and ``x*2-1`` out.
      * STREAMING CACHE: ``CausalTAE`` streaming keeps the per-block cache in
        an EXTERNAL list; ``Wan2_1_VAE`` keeps it INTERNAL and re-arms it when
        ``is_first_chunk=True``.  We hold the enc/dec caches on the adapter
        and rebuild them on ``is_first_chunk=True`` to reproduce the internal
        semantics exactly.

    Temporal convention matches ``Wan2_1_VAE`` (causal, first-frame aligned):
      ``T_pix = 4 * T_lat - 3`` (== ``CausalTAE``'s ``T_lat = n+1 <-> 4n+1``).
    """

    def __init__(self,
                 vae_pth: str,
                 z_dim: int = 16,
                 dtype=torch.float,
                 device="cuda",
                 **causal_tae_kwargs):
        self.dtype = dtype
        self.device = device
        # No mean/std normalisation: the CausalTAE latent space is
        # self-consistent (its own decoder inverts its own encoder), so we
        # expose a no-op ``scale`` only for attribute parity with Wan2_1_VAE.
        self.scale = [0.0, 1.0]

        self.model = CausalTAE(
            latent_channels=z_dim, **causal_tae_kwargs
        ).eval().requires_grad_(False)
        if vae_pth is not None:
            sd = torch.load(vae_pth, map_location="cpu", weights_only=True)
            # tolerate a wrapping dict (e.g. {"model": ...} / {"state_dict": ...})
            if isinstance(sd, dict):
                for key in ("state_dict", "model", "module"):
                    if key in sd and isinstance(sd[key], dict):
                        sd = sd[key]
                        break
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                import logging
                logging.info(
                    f"CausalTAEVAE: loaded {vae_pth} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")
        self.model = self.model.to(device)

        # streaming caches (internal, mirroring Wan2_1_VAE's internal state).
        self._enc_cache = None
        self._dec_cache = None

    # ---- layout / range helpers -------------------------------------------

    @staticmethod
    def _to_ntchw_01(clip):
        """``[C, T, H, W]`` in [-1,1]  ->  ``[1, T, C, H, W]`` in [0,1]."""
        x = clip.permute(1, 0, 2, 3).contiguous()   # [T, C, H, W]
        x = x.unsqueeze(0)                           # [1, T, C, H, W]
        return x.mul(0.5).add(0.5)                   # [-1,1] -> [0,1]

    @staticmethod
    def _from_ntchw_01(x):
        """``[1, T, C, H, W]`` in [0,1]  ->  ``[C, T, H, W]`` in [-1,1]."""
        x = x.squeeze(0)                             # [T, C, H, W]
        x = x.permute(1, 0, 2, 3).contiguous()       # [C, T, H, W]
        return x.mul(2.0).sub(1.0).clamp_(-1, 1)     # [0,1] -> [-1,1]

    @staticmethod
    def _lat_to_ntchw(z):
        """latent ``[C, T, H, W]``  ->  ``[1, T, C, H, W]`` (no range change)."""
        return z.permute(1, 0, 2, 3).contiguous().unsqueeze(0)

    @staticmethod
    def _lat_from_ntchw(z):
        """latent ``[1, T, C, H, W]``  ->  ``[C, T, H, W]``."""
        return z.squeeze(0).permute(1, 0, 2, 3).contiguous()

    # ---- one-shot ---------------------------------------------------------

    def encode(self, videos):
        """``videos``: list of ``[3, T_pix, H, W]`` in [-1,1] ->
        list of latent ``[16, T_lat, H/8, W/8]``.

        Uses the STREAMING encode (``encode_video_chunked``: the 1,4,4,4,...
        chunk schedule with a fresh cache per clip) rather than a one-shot
        ``encode_video`` -- numerically identical, but bounds peak memory."""
        with torch.amp.autocast('cuda', dtype=self.dtype), torch.no_grad():
            out = []
            for u in videos:
                x = self._to_ntchw_01(u).to(self.device)
                z = self.model.encode_video_chunked(x, show_progress_bar=False)
                out.append(self._lat_from_ntchw(z).float())
            return out

    def decode(self, zs):
        """``zs``: list of latent ``[16, T_lat, H, W]`` ->
        list of pixel ``[3, T_pix, H*8, W*8]`` in [-1,1].

        Uses the STREAMING decode (``decode_video_streamed``: one latent
        chunk at a time with a fresh cache per clip) rather than a one-shot
        ``decode_video`` -- numerically identical, but bounds peak memory."""
        with torch.amp.autocast('cuda', dtype=self.dtype), torch.no_grad():
            out = []
            for u in zs:
                z = self._lat_to_ntchw(u).to(self.device)
                x = self.model.decode_video_streamed(z, show_progress_bar=False)
                out.append(self._from_ntchw_01(x).float())
            return out

    # ---- streaming --------------------------------------------------------

    def encode_chunk(self, videos, is_first_chunk: bool = False):
        """Streaming encode.  Same list I/O as :meth:`encode`.  The adapter
        holds the encoder cache internally and re-arms it when
        ``is_first_chunk=True`` (mirroring ``Wan2_1_VAE.encode_chunk``)."""
        if is_first_chunk:
            self._enc_cache = self.model.init_encode_cache()
        with torch.amp.autocast('cuda', dtype=self.dtype), torch.no_grad():
            out = []
            for u in videos:
                x = self._to_ntchw_01(u).to(self.device)
                z = self.model.encode_chunk(
                    x, self._enc_cache, is_first_chunk=is_first_chunk)
                out.append(self._lat_from_ntchw(z).float())
            return out

    def decode_chunk(self, zs, is_first_chunk: bool = False):
        """Streaming decode.  Same list I/O as :meth:`decode`.  The adapter
        holds the decoder cache internally and re-arms it when
        ``is_first_chunk=True`` (mirroring ``Wan2_1_VAE.decode_chunk``)."""
        if is_first_chunk:
            self._dec_cache = self.model.init_decode_cache()
        with torch.amp.autocast('cuda', dtype=self.dtype), torch.no_grad():
            out = []
            for u in zs:
                z = self._lat_to_ntchw(u).to(self.device)
                x = self.model.decode_chunk(
                    z, self._dec_cache, is_first_chunk=is_first_chunk)
                out.append(self._from_ntchw_01(x).float())
            return out
