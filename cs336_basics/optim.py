from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 999),
        eps: float = 1e-8,
        weight_decay: float | None = None,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(
        self,
        closure: Optional[Callable] = None,
    ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m1 = state.get("m1", 0)
                m2 = state.get("m2", 0)
                grad = p.grad.data
                m1 = beta1 * m1 + (1 - beta1) * grad
                m2 = beta2 * m2 + (1 - beta2) * (grad**2)
                alpha_t = (
                    lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                )
                p.data -= alpha_t * m1 / (torch.sqrt(m2) + epsilon)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m1"] = m1
                state["m2"] = m2
        return loss


def adamW_adapter(
    params,
    lr: float = 1e-3,
    beta1=0.9,
    beta2=0.999,
    eps: float = 1e-8,
    weight_decay: float | None = None,
):
    return AdamW(params, lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)


def get_lr_cosine_sheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it <= warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    if it <= cosine_cycle_iters:
        return min_learning_rate + (1 / 2) * (
            1
            + math.cos(
                (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
            )
        ) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


class CosineLRSheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch, "deprecated")

    def get_lr(self) -> float:
        pass
