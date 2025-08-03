import torch
from collections.abc import Iterable
import numpy as np
from typing import Tuple
from jaxtyping import Int


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


class SequeunceDataset(torch.utils.data.Dataset):

    def __init__(self, file_path: str, sequence_length: int):
        super().__init__()
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.fd = np.memmap(file_path, dtype=np.int16, mode="r")
        self.length = (self.fd.shape[0] - 1) // self.sequence_length

    def __getitem__(
        self, index: int
    ) -> Tuple[Int[torch.Tensor, "seq"], Int[torch.Tensor, "seq"]]:
        data = self.fd[
            index * self.sequence_length : (index + 1) * self.sequence_length
        ]
        predictions = self.fd[
            (index * self.sequence_length)
            + 1 : ((index + 1) * self.sequence_length)
            + 1
        ]
        return torch.from_numpy(data), torch.from_numpy(predictions)

    def __len__(self):
        return self.length
