import math

import torch
import torch.optim as optim

def get_optimizer_and_lr_schedule(params, opt_config, lr_config):
    optimizer, lr = _get_optimizer(params, opt_config)
    lr_schedule = _get_lr_schedule(optimizer, lr, lr_config)
    return optimizer, lr_schedule

def get_optimizer(params, opt_config, selective_weight_decay=True):
    assert "opt_type" in opt_config
    opt_type = opt_config["opt_type"]
    params = filter(lambda p: p.requires_grad, params)
    if opt_type == "sgd":
        lr = opt_config.get("lr", 0.001)
        momentum = opt_config.get("momentum", 0.0)
        weight_decay = opt_config.get("weight_decay", 0.0)
        nesterov = opt_config.get("nesterov", False)

        if selective_weight_decay and weight_decay > 0.0:
            params = _apply_selective_weight_decay(params, weight_decay)
        optimizer = optim.SGD(params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    elif opt_type == "adamw":
        lr = opt_config.get("lr", 0.001)
        betas = opt_config.get("betas", [0.9, 0.999])
        eps = opt_config.get("eps", 1e-08)
        weight_decay = opt_config.get("weight_decay", 0.01)
        if selective_weight_decay and weight_decay > 0.0:
            params = _apply_selective_weight_decay(params, weight_decay)
        optimizer = optim.AdamW(params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=True
        )
    elif opt_type == "delayed_nesterov":
        lr = opt_config["lr"]
        beta = opt_config["beta"]
        c = opt_config.get("c", 0.0)
        buffer_size = opt_config.get("buffer_size", 1)
 
        optimizer = DelayedNesterovOptimizer(params,
            lr=lr,
            beta=beta,
            c=c,
            buffer_size=buffer_size
        )
    else:
        raise f"Unsupported opt_type={opt_type}"

    return optimizer

def _apply_selective_weight_decay(params, weight_decay):
    # Biases and LayerNorm parameters will not be decayed.
    decay_params = []
    nodecay_params = []
    for p in params:
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    return param_groups

# Implement Delayed Nesterov Update in https://arxiv.org/pdf/2401.09135.
class DelayedNesterovOptimizer:

    def __init__(self, params, lr, beta, c=0.0, buffer_size=1):
        self._params = list(params)
        self._lr = lr                   # Learning rate (initial value)
        self._beta = beta               # Momentum decay
        self._c = c                     # Momentum activation
        self._buffer_size = buffer_size # Buffer size (N in the pseudocode)
        self._t = 0                     # Step counter
        self._m = [torch.zeros_like(p) for p in self._params]  # Initial momentum m_0
        self._delta = [torch.zeros_like(p) for p in self._params]  # Aggregated gradient Δ

        # Create a parameter group structure to be compatible with LR scheduler
        self.param_groups = [{'params': self._params, 'lr': self._lr}]

    def step(self):
        # Accumulate gradients
        for i, p in enumerate(self._params):
            self._delta[i] += p.grad

        # Apply delayed Nesterov update
        if (self._t + 1) % self._buffer_size == 0:
            for i, param in enumerate(self._params):
                self._m[i] = self._beta * self._m[i] + self._delta[i] / self._buffer_size
                param.data -= self.param_groups[0]['lr'] * (
                    (1 - self._c * self._buffer_size + self._c) * self._beta * self._m[i] + param.grad / self._buffer_size
                )
                self._delta[i].zero_()  # Reset Δ after update
        else:
            # Apply regular update without momentum update
            for i, param in enumerate(self._params):
                param.data -= self.param_groups[0]['lr'] * (self._c * self._beta * self._m[i] + param.grad / self._buffer_size)

        # Increment time step
        self._t += 1

    def zero_grad(self):
        # Reset gradients to zero
        for param in self._params:
            if param.grad is not None:
                param.grad.zero_()

def get_lr_schedule(optimizer, lr_config, t_curr):
    if lr_config is None or "lr_type" not in lr_config:
        return DummyLrSchedule()

    lr_type = lr_config["lr_type"]
    if lr_type == "cosine_decay_after_linear_warmup":
        max_lr = lr_config["max_lr"]
        if "min_lr" in lr_config:
            min_lr = lr_config["min_lr"]
        else:
            min_lr = 0.1 * max_lr
        t_warmup = lr_config["t_warmup"]
        t_max = lr_config["t_max"]

        return CosineDecayAfterLinearWarmupLrSchedule(optimizer, max_lr, min_lr, t_warmup, t_max, t_curr)
    else:
        raise f"Unsupported lr_type={lr_type}"

class CosineDecayAfterLinearWarmupLrSchedule:
    def __init__(self, optimizer, max_lr, min_lr, t_warmup, t_max, t_curr=0):
        self._optimizer = optimizer
        self._max_lr = max_lr
        self._min_lr = min_lr
        self._t_warmup = t_warmup
        self._t_max = t_max
        self._t_curr = t_curr if t_curr is not None else 0

    def set_current_step(self, curr_step):
        if curr_step is not None:
            self._t_curr = curr_step
    
    def curr_lr(self):
        if self._t_curr <= self._t_warmup:
            lr = self._max_lr * (self._t_curr / self._t_warmup)
        elif self._t_curr <= self._t_max:
            p = (self._t_curr - self._t_warmup) / (self._t_max - self._t_warmup)
            lr = self._min_lr + (self._max_lr - self._min_lr) * 0.5 * (1 + math.cos(math.pi * p))
        else:
            lr = self._min_lr

        return lr

    def step(self):
        self._t_curr += 1
        lr = self.curr_lr()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class DummyLrSchedule:
    def __init__(self):
        pass
    
    def set_current_step(self, curr_step):
        pass

    def step(self):
        pass
