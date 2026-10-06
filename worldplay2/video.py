from pathlib import Path

import imageio
import torch


def save_video(tensor: torch.Tensor, path: str | Path, fps: int) -> None:
    """Save ``[3,T,H,W]`` video in ``[-1,1]``."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = (
        tensor.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .byte()
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    imageio.mimsave(output, frames, fps=fps, codec="libx264", quality=8)
