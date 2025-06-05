import json
import sys
import time

import numpy as np
import ray
import torch

from halos.simulation.common import (
    get_arg_parser,
    init_workers,
    shutdown,
)
from halos.ps import ParameterServer
from halos.utils import (
    checkpoint_model,
    get_a2a_comm_time_ms,
    get_compute_time_ms,
    get_slowest_inter_region_network_bandwidth,
    get_result_dir,
    save_results,
    ProgressTracker,
    TrainingDataDistributor
)

@ray.remote
class DilocoSimulator(object):

    def __init__(
        self,
        result_dir,
        exp_name,
        local_step_time_ms,
        global_batch_size,
        total_steps,
        model_name,
        outer_opt_config,
        network_bandwidths,
        val_interval_steps,
        checkpointing_interval_steps,
        worker_handles
    ):
        self._worker_handles = worker_handles
        self._training_data_distributor = TrainingDataDistributor(
            global_batch_size=global_batch_size,
            worker_configs=[h.worker_config for h in worker_handles],
            is_sync_training=True
        )

        self._result_dir = result_dir
        self._progress_tracker = ProgressTracker(
            exp_name,
            val_interval_steps,
            checkpointing_interval_steps,
            self
        )

        self._ps = ParameterServer(model_name, outer_opt_config, total_steps)
        self._pgrads_buffer = [torch.zeros_like(p.data) for p in self._ps.model.parameters()]

        self._time = 0.0
        self._per_round_compute_time = max(
            get_compute_time_ms(
                h.worker_config.num_local_steps,
                h.worker_config.worker_speed,
                local_step_time_ms
            ) 
            for h in worker_handles
        )
        comm_bytes = sum(p.data.element_size() * p.data.numel() for p in self._ps.model.parameters())
        slowest_bandwidth = get_slowest_inter_region_network_bandwidth(network_bandwidths)
        self._per_round_comm_time = get_a2a_comm_time_ms(comm_bytes, len(worker_handles), slowest_bandwidth)
        self._per_round_time = self._per_round_compute_time + self._per_round_comm_time

        print(f"DiLoCo per round time={self._per_round_time:.2f}ms, " + 
            f"compute={self._per_round_compute_time:.2f}ms, " +
            f"comm={self._per_round_comm_time:.2f}ms", file=sys.stderr)

    def run(self):
        while not self._ps.training_done:
            self._run_round()

        model = self._ps.model
        histories = self._ps.pop_train_histories() + \
            self._progress_tracker.pop_val_histories()

        self._progress_tracker.shutdown()

        return model, histories

    def _run_round(self):
        futures = []
        for worker_handle in self._worker_handles:
            start_idx, end_idx, start_steps = self._training_data_distributor.get_batch_indexes(
                worker_handle.worker_config)
            futures.append(worker_handle.run_local_steps(
                self._ps.model,
                start_idx,
                end_idx,
                start_steps
            ))

        num_local_steps_history, loss_history = self._aggregate_results(futures)

        self._time += self._per_round_time
        cur_time = self._time
        loss_avg = np.mean(loss_history).item()
        self._ps.update_model(
            self._pgrads_buffer,
            num_local_steps_history,
            loss_history,
            cur_time,
            loss_avg
        )

        self._progress_tracker.add_train_losses(
            num_local_steps_history,
            loss_history,
            cur_time,
            loss_avg)

    def _aggregate_results(self, futures):
        for pgb in self._pgrads_buffer:
            pgb.zero_()

        results = [None] * len(futures)
        pending_futures = list(enumerate(futures))
        while pending_futures:
            ready_futures, remaining_futures = ray.wait([f[1] for f in pending_futures], num_returns=1)
            for ready in ready_futures:
                for idx, (original_idx, future) in enumerate(pending_futures):
                    if future == ready:
                        result = ray.get(ready)
                        results[original_idx] = result
                        for pgb, pg in zip(self._pgrads_buffer, result[0]):
                            pgb += pg
                        pending_futures.pop(idx)
                        break

        for pgb in self._pgrads_buffer:
            pgb /= len(futures)

        num_local_steps_history = []
        loss_history = []
        for _, steps, losses in results:
            num_local_steps_history.extend(steps)
            loss_history.extend(losses)

        return num_local_steps_history, loss_history

    def on_val_interval(self, num_trained_steps):
        val_loss = ray.get(self._worker_handles[0].eval_val_loss(self._ps.model))
        self._progress_tracker.add_val_loss(self._time, val_loss)

    def on_checkpointing_interval(self, num_trained_steps):
        checkpoint_model(self._result_dir, self._ps.model, num_trained_steps)

def main(args):
    worker_handles, network_bandwidths = init_workers(args)
    result_dir = get_result_dir(args.result_dir)

    simulator = DilocoSimulator.remote(
        result_dir,
        args.exp_name,
        args.local_step_time_ms,
        args.global_batch_size,
        args.total_steps,
        args.model_name,
        json.loads(args.outer_opt_config),
        network_bandwidths,
        args.val_interval_steps,
        args.checkpointing_interval_steps,
        worker_handles
    )

    model, histories = ray.get(simulator.run.remote())
    save_results(result_dir, args, model, *histories)
    shutdown()

if __name__ == "__main__":
    parser = get_arg_parser()

    parser.add_argument('--outer_opt_config', type=str, required=True)
    args = parser.parse_args()

    main(args)
