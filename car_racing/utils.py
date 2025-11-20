import torch


def scale_input(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).float() / 127.5 - 1  # Normalize and reshape

def rescale_output(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1) * 127.5).permute(0, 2, 3, 1).byte()  # Denormalize and reshape