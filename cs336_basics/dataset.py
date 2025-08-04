import torch
import numpy as np
from typing import Tuple
from jaxtyping import Int


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, file_path: str, sequence_length: int):
        super().__init__()
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.fd = np.memmap(file_path, dtype=np.uint16, mode="r")
        print(file_path, self.fd[0])
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
        return torch.from_numpy(data.copy()).to(torch.long), torch.from_numpy(predictions.copy()).to(
            torch.long
        )

    def __len__(self):
        return self.length
