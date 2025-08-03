import torch
from collections.abc import Iterable
from jaxtyping import Float
from typing import BinaryIO, IO
import os
import yaml
from dataclasses import dataclass
import argparse
from cs336_basics.layers import TransformerLM
from cs336_basics.optim import adamW_adapter
from cs336_basics.configuration import TrainingConfiguration, TrainingSchema
from pydantic import ValidationError
from configuration_engine.logging import (
    YamlLogger,
    CSVLogger,
    YamlFileLogger,
    CSVFileLogger,
)
from uuid import uuid4
import optuna
from cs336_basics.dataset import SequenceDataset
from cs336_basics.loss import CELosss
import time


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
) -> int:
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


def train_loop(
    config: TrainingConfiguration,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    train_dataset: SequenceDataset,
    valid_dataset: SequenceDataset,
    metric_logger: CSVLogger,
    run_prefix: str,
    iter: int,
    training_parameters,
) -> float:
    best_metric: float = 100000000

    batch_size: int = training_parameters("batch_size", 0)
    # single_batch: int = training_parameters.get("single_batch", False)
    tokens_processed: int = training_parameters.get("tokens_processed", 0)
    iters_checkpoint: int = training_parameters.get("iters_checkpoint", 1000)

    context_length: int = model.context_length
    max_iters = tokens_processed // (context_length * batch_size)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset)
    loss_fn = CELosss()

    start = time.time()
    while iter < max_iters:
        iter += 1
        valid_loss_total = 0
        train_loss_total = 0
        for data, labels in train_dataloader:
            model.train(True)
            logits: Float[torch.Tensor, "... seq vocab_size"] = model(data)
            loss = loss_fn(logits, labels)
            train_loss = loss.mean().backward()
            train_loss_total += train_loss
            model.train(False)
        # if iter % iters_validate:
        for data, labels in valid_dataloader:
            logits: Float[torch.Tensor, "... seq vocab_size"] = model(data)
            loss = loss_fn(logits, labels)
            valid_loss = loss.mean().backward()
            valid_loss_total += valid_loss
        end = time.time()
        metric_logger.log(
            {
                "iter": iter,
                "time": end - start,
                "train_loss": train_loss_total / len(train_dataloader),
                "valid_loss": valid_loss_total / len(valid_dataloader),
            }
        )
        if valid_loss_total / len(valid_dataloader) < best_metric:
            best_metric = valid_loss_total / len(valid_dataloader)

        if iter % iters_checkpoint:
            save_checkpoint(
                model,
                optimizer=optimizer,
                iteration=iter,
                out=os.path.join(config.metadata.output_path, f"{run_prefix}-{iter}"),
            )
    return best_metric


def objective(
    trial: optuna.Trial,
    config: TrainingConfiguration,
    schema: TrainingSchema,
    config_logger: YamlLogger,
    metric_logger: CSVLogger,
    run_prefix: str,
):
    _, safe_tuner_parameters = config.construct_tuner_parameters()
    training_parameters, safe_training_parameters = config.suggest_training_params(
        trial
    )
    additional_parameters, _ = config.construct_additional_params()
    model_parameters, safe_model_parameters = config.suggest_model_params(trial)
    optimizer_parameters, safe_optimizer_parameters = config.suggest_optimizer_params(
        trial
    )
    model = TransformerLM(**model_parameters)
    optimizer = adamW_adapter(**optimizer_parameters)
    if "checkpoint_path" in additional_parameters:
        iter = load_checkpoint(
            additional_parameters["checkpoint_path"], model, optimizer
        )
    else:
        iter = 0
    config_logger.log(
        {
            "metadata": config.metadata.model_dump(),
            "train_dataset": schema.train_dataset.model_dump(),
            "validation_dataset": schema.validation_dataset.model_dump(),
            "tuner_parameters": safe_tuner_parameters,
            "additional_parameters": {
                **additional_parameters,
            },
            "model_parameters": safe_model_parameters,
            "training_parameters": safe_training_parameters,
            "optimizer_parameters": safe_optimizer_parameters,
        }
    )
    best_val = train_loop(
        config=config,
        model=model,
        optimizer=optimizer,
        train_dataset=config.training_dataset,
        valid_dataset=config.validation_datset,
        metric_logger=metric_logger,
        run_prefix=run_prefix,
        iter=iter,
        training_parameters=training_parameters,
    )
    return best_val


def experimenting():
    """
    Dostaneme hyperparemtry v yaml souboru.
    """
    run_id = str(uuid4())
    args = parse_args()
    try:
        with open(args.config_path) as config_file:
            conf_dict = yaml.safe_load(config_file)
        schema = TrainingSchema(**conf_dict)
        config = schema.build()
        config_logger = YamlFileLogger(
            os.path.join(
                config.metadata.output_path, f"{config.metadata.name}-{run_id}.yaml"
            )
        )
        metric_logger = CSVFileLogger(
            os.path.join(
                config.metadata.output_path, f"{config.metadata.name}-{run_id}.csv"
            )
        )
        tuner_paramaters, _ = config.construct_tuner_parameters()
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(
                trial,
                config,
                schema,
                config_logger,
                metric_logger,
                f"{config.metadata.name}-{run_id}",
            ),
            **tuner_paramaters,
        )
    except OSError as e:
        print(e)
    except ValidationError as e:
        print(e)
