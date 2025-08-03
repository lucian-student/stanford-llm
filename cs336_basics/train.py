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
from cs336_basics.configuration import TrainingConfiguration, TrainingSchema
from pydantic import ValidationError
from configuration_engine.logging import YamlLogger, CSVLogger
from uuid import uuid4


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
    config_path: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", required=True)
    args = parser.parse_args()
    return TrainingArguments(**vars(args))


def experimenting():
    """
    Dostaneme hyperparemtry v yaml souboru.
    """
    args = parse_args()
    try:
        with open(args.config_path) as config_file:
            conf_dict = yaml.safe_load(config_file)
            schmema = TrainingSchema(**conf_dict)
            config = schmema.build()
    except OSError as e:
        print(e)
    except ValidationError as e:
        print(e)
