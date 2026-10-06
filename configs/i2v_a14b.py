from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class I2VA14BConfig:
    text_len: int = 512
    t5_dtype: torch.dtype = torch.bfloat16
    param_dtype: torch.dtype = torch.bfloat16
    num_train_timesteps: int = 1000
    boundary: float = 0.9
    sample_shift: float = 5.0
    sample_guide_scale: tuple[float, float] = (3.5, 3.5)
    sample_neg_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"
        "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
        "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
        "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
        "三条腿，背景人很多，倒着走"
    )
    t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"
    vae_checkpoint: str = "Wan2.1_VAE.pth"
    low_noise_checkpoint: str = "low_noise_model"
    high_noise_checkpoint: str = "high_noise_model"
    vae_stride: tuple[int, int, int] = (4, 8, 8)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    sample_fps: int = 16


I2V_A14B_CONFIG = I2VA14BConfig()
