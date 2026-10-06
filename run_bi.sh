#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Visual self-attention backend: flash (default) or sage.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sage}"
YAW_ROTATION_SPEED_DEG="${YAW_ROTATION_SPEED_DEG:-3.0}"
PITCH_ROTATION_SPEED_DEG="${PITCH_ROTATION_SPEED_DEG:-3.0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
GUIDE_SCALE="${GUIDE_SCALE:-3.5}"

DISABLE_COMPILED_QK_ROPE="${DISABLE_COMPILED_QK_ROPE:-false}"
DISABLE_CONDITIONING_CACHE="${DISABLE_CONDITIONING_CACHE:-false}"
DISABLE_SELF_QKV_FUSION="${DISABLE_SELF_QKV_FUSION:-false}"
DISABLE_COMPILED_BLOCK_GLUE="${DISABLE_COMPILED_BLOCK_GLUE:-false}"
DISABLE_COMPILED_FFN="${DISABLE_COMPILED_FFN:-false}"
USE_DIT_FSDP="${USE_DIT_FSDP:-false}"
USE_T5_FSDP="${USE_T5_FSDP:-false}"
ENABLE_FP8_FFN="${ENABLE_FP8_FFN:-false}"
FP8_START_BLOCK="${FP8_START_BLOCK:-0}"
FP8_END_BLOCK="${FP8_END_BLOCK:-39}"
ENABLE_FP8_QKV="${ENABLE_FP8_QKV:-false}"
FP8_QKV_START_BLOCK="${FP8_QKV_START_BLOCK:-0}"
FP8_QKV_END_BLOCK="${FP8_QKV_END_BLOCK:-39}"

CKPT_DIR="${CKPT_DIR:-/path/to/model}"
LOW_NOISE_CKPT="${LOW_NOISE_CKPT:-/path/to/model/low_noise_model.safetensors}"
HIGH_NOISE_CKPT="${HIGH_NOISE_CKPT:-/path/to/model/high_noise_model.safetensors}"

INPUT_JSON="${INPUT_JSON:-/path/to/input.json}"

OUTPUT_PATH="${OUTPUT_PATH:-./outputs/bi}"

run_case() {
  local json_path="$1"
  local output_dir="${OUTPUT_PATH}"
  local qk_rope_args=()
  if [[ "${DISABLE_COMPILED_QK_ROPE}" == "true" ]]; then
    qk_rope_args+=(--disable_compiled_qk_rope)
  fi
  local conditioning_args=()
  if [[ "${DISABLE_CONDITIONING_CACHE}" == "true" ]]; then
    conditioning_args+=(--disable_conditioning_cache)
  fi
  local qkv_fusion_args=()
  if [[ "${DISABLE_SELF_QKV_FUSION}" == "true" ]]; then
    qkv_fusion_args+=(--disable_self_qkv_fusion)
  fi
  local compiled_region_args=()
  if [[ "${DISABLE_COMPILED_BLOCK_GLUE}" == "true" ]]; then
    compiled_region_args+=(--disable_compiled_block_glue)
  fi
  if [[ "${DISABLE_COMPILED_FFN}" == "true" ]]; then
    compiled_region_args+=(--disable_compiled_ffn)
  fi
  local fsdp_args=()
  if [[ "${USE_DIT_FSDP}" == "true" ]]; then
    fsdp_args+=(--dit_fsdp)
  fi
  if [[ "${USE_T5_FSDP}" == "true" ]]; then
    fsdp_args+=(--t5_fsdp)
  fi
  local fp8_args=()
  if [[ "${ENABLE_FP8_FFN}" == "true" || "${ENABLE_FP8_QKV}" == "true" ]]; then
    if [[ "${USE_DIT_FSDP}" == "true" ]]; then
      echo "DiT FP8 requires USE_DIT_FSDP=false" >&2
      return 2
    fi
    fp8_args+=(--offload_model false)
  fi
  if [[ "${ENABLE_FP8_FFN}" == "true" ]]; then
    fp8_args+=(
      --fp8_ffn
      --fp8_start_block "${FP8_START_BLOCK}"
      --fp8_end_block "${FP8_END_BLOCK}"
    )
  fi
  if [[ "${ENABLE_FP8_QKV}" == "true" ]]; then
    fp8_args+=(
      --fp8_qkv
      --fp8_qkv_start_block "${FP8_QKV_START_BLOCK}"
      --fp8_qkv_end_block "${FP8_QKV_END_BLOCK}"
    )
  fi

  torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    generate.py \
    --inference_mode bi \
    --attention_backend "${ATTENTION_BACKEND}" \
    --ckpt_dir "${CKPT_DIR}" \
    --low_noise_mem_ckpt "${LOW_NOISE_CKPT}" \
    --high_noise_mem_ckpt "${HIGH_NOISE_CKPT}" \
    --input_json "${json_path}" \
    --yaw_rotation_speed_deg "${YAW_ROTATION_SPEED_DEG}" \
    --pitch_rotation_speed_deg "${PITCH_ROTATION_SPEED_DEG}" \
    --height 448 \
    --width 832 \
    --chunk_length 32 \
    --sink_size 1 \
    --temporal_size 1 \
    --sample_steps "${SAMPLE_STEPS}" \
    --sample_shift 5.0 \
    --sample_guide_scale "${GUIDE_SCALE}" \
    --seed 42 \
    "${qk_rope_args[@]}" \
    "${conditioning_args[@]}" \
    "${qkv_fusion_args[@]}" \
    "${compiled_region_args[@]}" \
    "${fsdp_args[@]}" \
    "${fp8_args[@]}" \
    --ulysses_size "${NPROC_PER_NODE}" \
    --output_dir "${output_dir}"
}

run_case "${INPUT_JSON}"
