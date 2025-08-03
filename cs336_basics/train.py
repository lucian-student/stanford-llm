import torch
from collections.abc import Iterable


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
):
    """
    To udělá aby norma byla rovná max_l2_norm, protože se to v podstatě vloží do každého elementu
    """
    norm_squared = 0
    for p in parameters:
        if p.grad is None:
            continue
        norm_squared += torch.pow(p.grad, 2).sum()
    norm = torch.sqrt(norm_squared)
    if norm > max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(max_l2_norm / (norm + eps))
