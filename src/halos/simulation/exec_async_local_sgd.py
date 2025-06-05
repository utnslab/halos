import json
import sys
import time

import numpy as np
import ray

from halos.simulation.common import (
    get_arg_parser,
    init_workers,
    shutdown,
    BaseAsyncSimulatior
)
from halos.ps import ParameterServer
from halos.utils import (
    checkpoint_model,
    get_result_dir,
    save_results,
    ProgressTracker
)

@ray.remote
class AsyncLocalSgdSimulator(BaseAsyncSimulatior):

    def __init__(
        self,
        result_dir,
        exp_name,
        local_step_time_ms,
        global_batch_size,
        total_steps,
        model_name,
        ps_opt_config,
        network_bandwidths,
        val_interval_steps,
        checkpointing_interval_steps,
        worker_handles
    ):
        super().__init__(
            local_step_time_ms=local_step_time_ms,
            global_batch_size=global_batch_size,
            network_bandwidths=network_bandwidths,
            worker_handles=worker_handles
        )
        self._result_dir = result_dir
        self._progress_tracker = ProgressTracker(
            exp_name,
            val_interval_steps,
            checkpointing_interval_steps,
            self
        )

        self._ps = ParameterServer(model_name, ps_opt_config, total_steps)
        self._worker_ids = [h.worker_id for h in worker_handles]

    def on_new_worker_pgrads(
        self,
        ps_id,
        worker_id,
        pgrads, 
        num_local_steps_history,
        loss_history
    ):
        cur_time = self.time
        loss_avg = np.mean(loss_history).item()

        self._ps.update_model(
            pgrads,
            num_local_steps_history,
            loss_history,
            cur_time,
            loss_avg
        )
        self._schedule_worker(worker_id)

        self._progress_tracker.add_train_losses(
            num_local_steps_history,
            loss_history,
            cur_time,
            loss_avg)

    def run(self):
        for worker_id in self._worker_ids:
            self._schedule_worker(worker_id)

    def _schedule_worker(self, worker_id):
        self.run_local_steps(
            ps_id=0,
            ps_region_id=0,
            worker_id=worker_id,
            start_model=self._ps.model
        )

    def training_done(self):
        return self._ps.training_done

    def get_histories(self):
        return self._ps.pop_train_histories() + \
            self._progress_tracker.pop_val_histories()

    def on_val_interval(self, num_trained_steps):
        val_loss = self.eval_val_loss(self._ps.model)
        self._progress_tracker.add_val_loss(self.time, val_loss)

    def on_checkpointing_interval(self, num_trained_steps):
        checkpoint_model(self._result_dir, self._ps.model, num_trained_steps)

    def get_model(self):
        return self._ps.model

    def shutdown(self):
        self._progress_tracker.shutdown()

def main(args):
    worker_handles, network_bandwidths = init_workers(args)
    result_dir = get_result_dir(args.result_dir)

    simulator = AsyncLocalSgdSimulator.remote(
        result_dir,
        args.exp_name,
        args.local_step_time_ms,
        args.global_batch_size,
        args.total_steps,
        args.model_name,
        json.loads(args.ps_opt_config),
        network_bandwidths,
        args.val_interval_steps,
        args.checkpointing_interval_steps,
        worker_handles
    )

    ray.get(simulator.run.remote())

    while not ray.get(simulator.training_done.remote()):
        time.sleep(10.0)

    histories = ray.get(simulator.get_histories.remote())
    model = ray.get(simulator.get_model.remote())
    save_results(result_dir, args, model, *histories)

    ray.get(simulator.shutdown.remote())
    shutdown()

if __name__ == "__main__":
    parser = get_arg_parser()

    parser.add_argument('--ps_opt_config', type=str, required=True)
    args = parser.parse_args()

    main(args)
