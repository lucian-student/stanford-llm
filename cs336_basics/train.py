import torch
from collections.abc import Iterable
import numpy as np
from typing import Tuple
from jaxtyping import Int
from typing import BinaryIO, IO
import os
import yaml
from dataclasses import dataclass
import argparse
from cs336_basics.layers import TransformerLM
from cs336_basics.optim import AdamW


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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_dict = model.state_dict()
    optimizer_dict = optimizer.state_dict()
    torch.save(
        {"model": model_dict, "optimizer": optimizer_dict, "iteration": iteration}, out
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    data = torch.load(src)
    if "model" in data:
        model.load_state_dict(data["model"])
    if "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    return data.get("iteration", 0)


@dataclass
class TrainingArguments:
    pass


def experimenting():
    """
    Dostaneme hyperparemtry v yaml souboru.
    """
    pass
