import torch
from typing import Callable, Dict, Union

import yaml
from cs336_basics.layers import TransformerLM
from cs336_basics.loss import CELoss
from cs336_basics.optim import AdamW
import argparse
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass


def benchmark():
    pass

def profile(
    callback: Callable,
    num_warmups: int = 1,
    profiler_config: Dict = {},
    table_config: Dict = {"sort_by": "cuda_time_total", "row_limit": 20},
):

    for _ in range(num_warmups):
        callback()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        **profiler_config,
    ) as prof:
        callback()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return prof.key_averages().table(**table_config)


class BenchmarkSchema(BaseModel):
    model_parameters: Dict[str, Union[int, float, str, bool]]


@dataclass
class BechmarkArguments:
    config_path: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", required=True)
    args = parser.parse_args()
    return BechmarkArguments(**vars(args))


def llm_benchmark_random(model: TransformerLM,optimizer:AdamW, device: torch.device):
    loss_fn = CELoss()
    total_loss = 0
    for _ in range(10):
        data = torch.randint(
            low=0, high=100, size=(32, model.context_length), dtype=torch.long, device=device
        )
        label = torch.randint(
            low=0, high=100, size=(32, model.context_length), dtype=torch.long, device=device
        )
        out = model(data)
        loss = loss_fn(out, label)
        loss = loss.mean()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss


def llm_benchmark():
    """
    Benchmarkuje rychlost tranformeru
    """
    try:
        args = parse_args()
        with open(args.config_path) as config:
            data = yaml.safe_load(config)
        schema = BenchmarkSchema(**data)
        model = TransformerLM(**schema.model_parameters)
        optimizer = AdamW(model.parameters())
        device = torch.device("cuda")
        model.to(device)
        model = torch.compile(model)
        print(profile(lambda: llm_benchmark_random(model,optimizer, device)))
    except OSError as e:
        print(e)
    except ValidationError as e:
        print(e)
