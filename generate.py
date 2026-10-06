#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import logging
import sys

import torch
from PIL import Image

from configs import I2V_A14B_CONFIG
from worldplay2.inputs import (
    initialize_runtime,
    load_inference_inputs,
)
from worldplay2.pipeline import InferenceMode, WorldPlay2Pipeline
from worldplay2.video import save_video


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="WorldPlay2 action-conditioned video inference"
    )
    source = parser.add_argument_group("input")
    source.add_argument("--input_json", required=True)
    source.add_argument(
        "--rotation_speed_deg",
        type=float,
        default=3.0,
        help="Fallback rotation degrees per latent for axes without an override.",
    )
    source.add_argument(
        "--yaw_rotation_speed_deg",
        type=float,
        help="Left/right rotation degrees per latent; defaults to rotation_speed_deg.",
    )
    source.add_argument(
        "--pitch_rotation_speed_deg",
        type=float,
        help="Up/down rotation degrees per latent; defaults to rotation_speed_deg.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--ckpt_dir", required=True)
    model.add_argument("--low_noise_mem_ckpt", required=True)
    model.add_argument("--high_noise_mem_ckpt", required=True)
    model.add_argument(
        "--inference_mode",
        choices=[mode.value for mode in InferenceMode],
        default=InferenceMode.BI.value,
    )
    model.add_argument("--vae_type", choices=["wan2_1", "causal_tae"], default="wan2_1")
    model.add_argument("--tae_ckpt")

    generation = parser.add_argument_group("generation")
    generation.add_argument("--height", type=int, default=448)
    generation.add_argument("--width", type=int, default=832)
    generation.add_argument("--chunk_length", type=int, default=4)
    generation.add_argument("--sink_size", type=int, default=1)
    generation.add_argument("--temporal_size", type=int, default=1)
    generation.add_argument("--sample_steps", type=int)
    generation.add_argument("--sample_shift", type=float)
    generation.add_argument("--sample_guide_scale", type=float)
    generation.add_argument("--sample_solver", choices=["unipc", "dpm++"], default="unipc")
    generation.add_argument("--negative_prompt", default="")
    generation.add_argument("--seed", type=int, default=-1)
    generation.add_argument("--output_dir", default="outputs")
    generation.add_argument(
        "--compile_cache_dir",
        default="~/.cache",
        help="Persistent torch.compile cache directory.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--offload_model", type=str2bool)
    runtime.add_argument(
        "--attention_backend",
        choices=["flash", "sage"],
        default="flash",
        help="Visual self-attention backend: FlashAttention or SageAttention. "
             "Text cross-attention and masked prefill keep their existing backends.",
    )
    runtime.add_argument("--ulysses_size", type=int, default=1)
    runtime.add_argument(
        "--disable_compiled_qk_rope",
        action="store_true",
        help="Use the eager QK RMSNorm + RoPE kernel.",
    )
    runtime.add_argument(
        "--disable_conditioning_cache",
        action="store_true",
        help="Disable text projection and cross-attention K/V caches.",
    )
    runtime.add_argument(
        "--disable_self_qkv_fusion",
        action="store_true",
        help="Keep separate self-attention Q/K/V projection GEMMs.",
    )
    runtime.add_argument(
        "--disable_compiled_block_glue",
        action="store_true",
        help="Use eager modulation and gated residual expressions.",
    )
    runtime.add_argument(
        "--disable_compiled_ffn",
        action="store_true",
        help="Use the eager Linear-GELU-Linear FFN module.",
    )
    runtime.add_argument(
        "--fp8_ffn",
        action="store_true",
        help="Use Transformer Engine FP8 E4M3 for selected FFN blocks. "
             "Requires DiT FSDP and model offload to be disabled.",
    )
    runtime.add_argument("--fp8_start_block", type=int, default=4)
    runtime.add_argument("--fp8_end_block", type=int, default=35)
    runtime.add_argument(
        "--fp8_qkv",
        action="store_true",
        help="Use Transformer Engine FP8 E4M3 for selected fused "
             "self-attention QKV projections.",
    )
    runtime.add_argument("--fp8_qkv_start_block", type=int, default=4)
    runtime.add_argument("--fp8_qkv_end_block", type=int, default=35)
    runtime.add_argument("--t5_fsdp", action="store_true")
    runtime.add_argument("--dit_fsdp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    inference = load_inference_inputs(
        args,
        default_guide_scale=I2V_A14B_CONFIG.sample_guide_scale,
    )
    runtime = initialize_runtime(args)
    mode = InferenceMode(inference.mode)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout)
            if runtime.rank == 0
            else logging.NullHandler()
        ],
        force=True,
    )

    pipeline = WorldPlay2Pipeline(
        config=I2V_A14B_CONFIG,
        checkpoint_dir=args.ckpt_dir,
        low_noise_mem_ckpt=args.low_noise_mem_ckpt,
        high_noise_mem_ckpt=args.high_noise_mem_ckpt,
        device_id=runtime.local_rank,
        rank=runtime.rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=args.ulysses_size > 1,
        attention_backend=args.attention_backend,
        compile_qk_rope=not args.disable_compiled_qk_rope,
        conditioning_cache=not args.disable_conditioning_cache,
        fuse_self_qkv=not args.disable_self_qkv_fusion,
        compile_block_glue=not args.disable_compiled_block_glue,
        compile_ffn=not args.disable_compiled_ffn,
        fp8_ffn=args.fp8_ffn,
        fp8_start_block=args.fp8_start_block,
        fp8_end_block=args.fp8_end_block,
        fp8_qkv=args.fp8_qkv,
        fp8_qkv_start_block=args.fp8_qkv_start_block,
        fp8_qkv_end_block=args.fp8_qkv_end_block,
        inference_mode=mode,
        vae_type=args.vae_type,
        tae_ckpt=args.tae_ckpt,
    )
    # FSDP construction can temporarily materialize full parameters. Clear
    # allocator cache so nvidia-smi reflects persistent usage rather than
    # stale reserved blocks from initialization.
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(
        "FSDP requested: t5=%s dit=%s; wrapper counts: t5=%d low=%d high=%d",
        args.t5_fsdp,
        args.dit_fsdp,
        pipeline.dtype_report["t5_fsdp_units"],
        pipeline.dtype_report["low_noise_fsdp_units"],
        pipeline.dtype_report["high_noise_fsdp_units"],
    )

    for item in inference.items:
        image = Image.open(item.image).convert("RGB")
        video = pipeline.generate_chunked(
            prompts=item.prompts,
            chunk_prompts=item.chunk_prompts,
            img=image,
            hr_action=item.actions.tensor,
            total_latent_frames=item.actions.latent_frames,
            chunk_length=args.chunk_length,
            shift=inference.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=inference.sample_steps,
            guide_scale=inference.guide_scale,
            n_prompt=args.negative_prompt,
            seed=runtime.seed,
            offload_model=runtime.offload_model,
            sink_size=args.sink_size,
            temporal_size=args.temporal_size,
            height=args.height,
            width=args.width,
        )
        if runtime.rank == 0 and video is not None:
            save_video(
                video,
                runtime.output_dir / f"{item.output_name}.mp4",
                I2V_A14B_CONFIG.sample_fps,
            )
    runtime.finish()


if __name__ == "__main__":
    main()
