import torch
from collections.abc import Iterable
from jaxtyping import Float
from typing import BinaryIO, IO
import os
import yaml
from dataclasses import dataclass
import argparse
from cs336_basics.layers import TransformerLM
from cs336_basics.optim import adamW_adapter, CosineLRSheduler
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
from cs336_basics.torch_utils import torch_setup
import time
import tqdm
import math


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
    return norm


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
    trial_number: int,
    config: TrainingConfiguration,
    model: TransformerLM,
    sheduler: CosineLRSheduler,
    optimizer: torch.optim.Optimizer,
    train_dataset: SequenceDataset,
    valid_dataset: SequenceDataset,
    metric_logger: CSVLogger,
    train_logger: CSVLogger,
    run_prefix: str,
    iter: int,
    training_parameters,
) -> float:
    """
    Důležitý je ukádáat parametry a optimizer state v 32flaot
    """
    best_metric: float = 100000000

    iters_checkpoint: int = training_parameters.get("iters_checkpoint", 5000)
    iters_validation: int = training_parameters.get("iters_validation", 2000)
    batch_size: int = training_parameters.get("batch_size", 0)
    tokens_processed: int = training_parameters.get("tokens_processed", 0)
    context_length: int = model.context_length
    # tokens_processed 32768000 <class 'int'> batch_size 32 <class 'int'> context_length 256 <class 'int'>
    max_iters = tokens_processed // (context_length * batch_size)
    print(
        "tokens_processed",
        tokens_processed,
        type(tokens_processed),
        "batch_size",
        batch_size,
        type(batch_size),
        "context_length",
        context_length,
        type(context_length),
    )
    print("max_iters: ", max_iters)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset)
    loss_fn = CELosss()
    if training_parameters.get("cuda", False):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    start = time.time()
    train_loss_total = 0
    batches = 0
    while iter < max_iters:
        for _, (data, labels) in tqdm.tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            batches+=1
            iter += 1
            if iter >= max_iters:
                break
            ##print("data and labels: ",data,labels)
            data_device: torch.Tensor = data.to(device)
            labels_device: torch.Tensor = labels.to(device)
            model.train(True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):  
                logits: Float[torch.Tensor, "... seq vocab_size"] = model(data_device)
            loss = loss_fn(logits, labels_device)
            train_loss = loss.mean()
            batch_perplexity = torch.exp(train_loss)
            train_loss.backward()
            norm = clip_gradients(
                model.parameters(), training_parameters.get("max_l2_norm", 1.0)
            )
            optimizer.step()
            sheduler.step()
            optimizer.zero_grad()
            train_loss_total += train_loss.item()
            model.train(False)
            train_logger.log(
                {
                    "iter": iter,
                    "trial_number": trial_number,
                    "train_loss": train_loss.item(),
                    "gradient_norm": norm.item(),
                    "batch_perplexity": batch_perplexity.item(),
                }
            )

            if iter % iters_validation == 0:
                valid_loss_total = 0
                for data, labels in valid_dataloader:
                    with torch.no_grad():
                        data_device: torch.Tensor = data.to(device)
                        labels_device: torch.Tensor = labels.to(device)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            logits: Float[torch.Tensor, "... seq vocab_size"] = model(
                                data_device
                            )
                        loss = loss_fn(logits, labels_device)
                        valid_loss = loss.mean()
                        valid_loss_total += valid_loss.item()
                end = time.time()
                metric_logger.log(
                    {
                        "iter": iter,
                        "trial_number": trial_number,
                        "time": end - start,
                        "train_loss": train_loss_total / batches,
                        "valid_loss": valid_loss_total / len(valid_dataloader),
                        "train_perplexity": math.exp(
                            train_loss_total / batches
                        ),
                        "valid_perplexity": math.exp(
                            valid_loss_total / len(valid_dataloader)
                        ),
                    }
                )
                batches = 0
                train_loss_total = 0
                if valid_loss_total / len(valid_dataloader) < best_metric:
                    best_metric = valid_loss_total / len(valid_dataloader)
            if iter % iters_checkpoint == 0:
                save_checkpoint(
                    model,
                    optimizer=optimizer,
                    iteration=iter,
                    out=os.path.join(
                        config.metadata.output_path,
                        f"{trial_number}-{run_prefix}-{iter}",
                    ),
                )
        save_checkpoint(
            model,
            optimizer=optimizer,
            iteration=iter,
            out=os.path.join(
                config.metadata.output_path, f"{trial_number}-{run_prefix}-{iter}"
            ),
        )
    return best_metric


def objective(
    trial: optuna.Trial,
    config: TrainingConfiguration,
    schema: TrainingSchema,
    config_logger: YamlLogger,
    metric_logger: CSVLogger,
    train_logger: CSVLogger,
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
    optimizer = adamW_adapter(model.parameters(), **optimizer_parameters)
    if "checkpoint_path" in additional_parameters:
        iter = load_checkpoint(
            additional_parameters["checkpoint_path"], model, optimizer
        )
    else:
        iter = 0

    batch_size: int = training_parameters.get("batch_size", 0)
    tokens_processed: int = training_parameters.get("tokens_processed", 0)
    context_length: int = model.context_length
    max_iters = tokens_processed // (context_length * batch_size)
    sheduler = CosineLRSheduler(
        optimizer,
        cosine_cycle_iters=max_iters
        * training_parameters.get("cosine_cycle_iters", 1.0),
        warmup_iters=training_parameters.get("warmup_iters", 500),
        min_lr=training_parameters.get("min_lr", 0.1)
        * optimizer_parameters.get("lr", 1e-3),
        last_epoch=iter - 1,
    )
    config_logger.log(
        {
            "metadata": config.metadata.model_dump(),
            "train_dataset": schema.train_dataset.model_dump(),
            "validation_dataset": schema.validation_dataset.model_dump(),
            "tuner_parameters": safe_tuner_parameters,
            "additional_parameters": {
                **additional_parameters,
                "trial_number": trial.number,
            },
            "model_parameters": safe_model_parameters,
            "training_parameters": safe_training_parameters,
            "optimizer_parameters": safe_optimizer_parameters,
        }
    )
    best_val = train_loop(
        trial_number=trial.number,
        config=config,
        model=model,
        optimizer=optimizer,
        sheduler=sheduler,
        train_dataset=config.training_dataset,
        valid_dataset=config.validation_datset,
        metric_logger=metric_logger,
        train_logger=train_logger,
        run_prefix=run_prefix,
        iter=iter,
        training_parameters=training_parameters,
    )
    return best_val


def experimenting():
    """
    Dostaneme hyperparemtry v yaml souboru.
    """
    torch_setup()
    run_id = str(uuid4())
    print(f"RUN_ID: {run_id}!")
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
        train_logger = CSVFileLogger(
            os.path.join(
                config.metadata.output_path,
                f"{config.metadata.name}-train-{run_id}.csv",
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
                train_logger,
                f"{config.metadata.name}-{run_id}",
            ),
            **tuner_paramaters,
        )
    except OSError as e:
        print(e)
    except ValidationError as e:
        print(e)
