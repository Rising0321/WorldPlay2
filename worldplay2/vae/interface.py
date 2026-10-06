from typing import Protocol

import torch


class StreamingVAE(Protocol):
    def encode(self, videos: list[torch.Tensor]) -> list[torch.Tensor]: ...
    def decode(self, latents: list[torch.Tensor]) -> list[torch.Tensor]: ...
    def encode_chunk(
        self, videos: list[torch.Tensor], is_first_chunk: bool = False
    ) -> list[torch.Tensor]: ...
    def decode_chunk(
        self, latents: list[torch.Tensor], is_first_chunk: bool = False
    ) -> list[torch.Tensor]: ...
