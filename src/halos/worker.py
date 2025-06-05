import copy
import math

import torch

from halos.opt import get_optimizer, get_lr_schedule

class Worker(object):

    def __init__(
        self,
        worker_config,
        model,
        device
    ):
        self._config = worker_config
        self._device = device

        self._num_trained_steps = 0
        self._model = model

        self._optimizer = None
        self._lr_schedule = None

        self._selective_weight_decay = True
        self._grad_clip = 1.0
        self._dtype = torch.float16
        self._amp_enabled = self._dtype == torch.float16 or self._dtype == torch.bfloat16

        self._scaler = torch.GradScaler("cuda", enabled=self._amp_enabled)
        self._amp_ctx = torch.autocast("cuda", enabled=self._amp_enabled, dtype=self._dtype)

    @property
    def config(self):
        return self._config

    def _init_model(self, start_model, start_steps):
        for p, new_p in zip(self._model.parameters(), start_model.parameters()):
            p.data.copy_(new_p.data)

        if self._optimizer is None:
            self._optimizer = get_optimizer(
                self._model.parameters(),
                self._config.opt_config,
                selective_weight_decay=self._selective_weight_decay
            )

            self._lr_schedule = get_lr_schedule(self._optimizer, self._config.lr_config, start_steps)
        else:
            self._load_optimizer_states()
            self._lr_schedule.set_current_step(start_steps)

        return self._model, self._optimizer, self._lr_schedule

    def _save_optimizer_states(self):
        self._optimizer_states_to_device("cpu")

    def _load_optimizer_states(self):
        self._optimizer_states_to_device(self._device)

    def _optimizer_states_to_device(self, device):
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def run_local_steps(self, start_model, start_steps, data_iterator):
        start_model_params = [p.data.clone() for p in start_model.parameters()]
        model, optimizer, lr_schedule = self._init_model(start_model, start_steps)

        n_steps = 0
        loss_sum = torch.tensor(0.0, device=self._device)
        num_local_steps_history = []
        loss_history = []

        num_local_steps = self._config.num_local_steps
        num_gradient_accumulation = self._config.num_gradient_accumulation
        loss_average_step = self._config.loss_average_step

        optimizer.zero_grad()
        model.train()
        for _ in range(num_local_steps):
            lr_schedule.step()
            for _ in range(num_gradient_accumulation):
                batch = next(data_iterator)
                batch = batch.to(self._device)
                with self._amp_ctx:
                    outputs = model(**batch)
                    loss = outputs.loss / num_gradient_accumulation

                self._scaler.scale(loss).backward()
                loss_sum += loss

            if self._grad_clip > 0.0:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)

            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad(set_to_none=True)

            n_steps += 1
            if n_steps % loss_average_step == 0:
                loss_history.append(loss_sum.item() / loss_average_step)
                num_local_steps_history.append(loss_average_step)
                loss_sum.zero_()

        if n_steps % loss_average_step != 0:
            remainder = n_steps % loss_average_step
            loss_history.append(loss_sum.item() / remainder)
            num_local_steps_history.append(remainder)
            last_loss = loss_history[-1]
        else:
            last_loss = loss_history[-1]

        self._num_trained_steps += n_steps
        self._save_optimizer_states()
        prads = [start_param - param.data.to("cpu") for start_param, param in zip(start_model_params, model.parameters())]
        return prads, num_local_steps_history, loss_history

    def eval_val_loss(self, model, data_iterator):
        for p, new_p in zip(self._model.parameters(), model.parameters()):
            p.data.copy_(new_p.data)

        num_batches = 0
        loss_sum = torch.tensor(0.0, device=self._device)
        with torch.no_grad():
            for batch in data_iterator:
                batch = batch.to(self._device)
                with self._amp_ctx:
                    outputs = self._model(**batch)
                loss = outputs.loss
                loss_sum += loss
                num_batches += 1

        return loss_sum.item() / num_batches
