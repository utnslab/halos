import numpy as np

from halos.model import load_pythia_model
from halos.opt import get_optimizer

class ParameterServer(object):

    def __init__(
        self,
        model_name,
        opt_config,
        total_steps=-1
    ):
        self._model = load_pythia_model(model_name)
        self._optimizer = get_optimizer(
            self._model.parameters(),
            opt_config
        )
        self._total_steps = total_steps

        self._num_local_steps_history = []
        self._loss_history = []
        self._time_history = []
        self._loss_over_time_history = []
        self._num_trained_steps = 0

    @property
    def model(self):
        return self._model
    
    @property
    def num_trained_steps(self):
        return self._num_trained_steps

    @property
    def training_done(self):
        if self._total_steps > 0 and self.num_trained_steps >= self._total_steps:
            return True
        return False

    def update_model(
        self,
        pgrads,
        num_local_steps_history,
        loss_history,
        cur_time,
        loss_avg
    ):
        if self.training_done:
            return

        self._num_local_steps_history.extend(num_local_steps_history)
        self._loss_history.extend(loss_history)
        self._time_history.append(cur_time)
        self._loss_over_time_history.append(loss_avg)

        self._num_trained_steps += sum(num_local_steps_history)

        self._model.train()
        self._optimizer.zero_grad()
        for p, g in zip(self._model.parameters(), pgrads):
            p.grad = g

        self._optimizer.step()

    def merge_model(self, merge_params, merge_weight):
        for cur_p, merge_p in zip(self._model.parameters(), merge_params):
            cur_p.data *= (1.0 - merge_weight)
            cur_p.data += merge_p * merge_weight

    def pop_train_histories(self):
        num_local_steps_history = self._num_local_steps_history
        loss_history = self._loss_history
        time_history = self._time_history
        loss_over_time_history = self._loss_over_time_history

        self._num_local_steps_history = []
        self._loss_history = []
        self._time_history = []
        self._loss_over_time_history = []

        return num_local_steps_history, \
            loss_history, \
            time_history, \
            loss_over_time_history
