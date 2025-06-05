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

class LocalParameterServer(object):
    
    def __init__(
        self,
        model_name,
        opt_config,
        model_merge_weight,
        local_updates_accumulation,
        lps_id,
        region_id,
        worker_ids,
        simulator
    ):
        assert local_updates_accumulation > 0
        self._ps = ParameterServer(model_name, opt_config)

        self._model_version = 0
        self._merged_model_version = 0
        self._merged_model_params = [p.data.clone() for p in self._ps.model.parameters()]

        self._model_merge_weight = model_merge_weight
        self._local_updates_accumulation = local_updates_accumulation
        self._lps_id = lps_id
        self._region_id = region_id
        self._worker_ids = worker_ids
        self._simulator = simulator
    
    @property
    def time(self):
        return self._simulator.time

    @property
    def region_id(self):
        return self._region_id

    def on_new_worker_pgrads(
        self,
        worker_id,
        pgrads,
        num_local_steps_history,
        loss_history,
        cur_time,
        loss_avg
    ):
        print(f"[Time={self.time:.2f}] LPS-{self._lps_id} (model_version={self._model_version})" + 
            f" Receive pgrads from worker_id={worker_id}", file=sys.stderr)
        self._ps.update_model(
            pgrads,
            num_local_steps_history,
            loss_history,
            cur_time,
            loss_avg
        )

        self._model_version += 1

        if self._model_version == self._merged_model_version + self._local_updates_accumulation:
            self._send_updates_to_gps()

        self._schedule_worker(worker_id)

    def on_new_global_model(self, global_params):
        print(f"[Time={self.time:.2f}] LPS-{self._lps_id} (model_version={self._model_version})" + 
            f" Receive global model", file=sys.stderr)
        self._ps.merge_model(global_params, self._model_merge_weight)
        for merged_param, param in zip(self._merged_model_params, self._ps.model.parameters()):
            merged_param.copy_(param.data)
        self._merged_model_version = self._model_version

    def _send_updates_to_gps(self):
        print(f"[Time={self.time:.2f}] LPS-{self._lps_id} (model_version={self._model_version})" + 
            f" Send updates to GPS", file=sys.stderr)
        updates = []
        for latest_merged_param, param in zip(
            self._merged_model_params,
            self._ps.model.parameters()
        ):
            updates.append(latest_merged_param - param.data)
        
        num_local_steps_history, loss_history, _, _ = self._ps.pop_train_histories()
        self._simulator.send_updates_to_gps(
            self._lps_id,
            updates,
            num_local_steps_history,
            loss_history)

    def _schedule_worker(self, worker_id):
        self._simulator.run_local_steps(
            ps_id=self._lps_id,
            ps_region_id=self._region_id,
            worker_id=worker_id,
            start_model=self._ps.model
        )

    def run(self):
        for worker_id in self._worker_ids:
            self._schedule_worker(worker_id)

@ray.remote
class HalosSimulator(BaseAsyncSimulatior):

    def __init__(
        self,
        result_dir,
        exp_name,
        local_step_time_ms,
        global_batch_size,
        total_steps,
        model_name,
        gps_opt_config,
        num_lps,
        num_workers_per_lps,
        lps_opt_config,
        model_merge_weight,
        local_updates_accumulation,
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

        self._gps = ParameterServer(model_name, gps_opt_config, total_steps)
        self._gps_region_id = 0

        self._lps_list = []
        worker_idx = 0
        for lps_id in range(num_lps):
            worker_configs_in_lps = [
                h.worker_config
                for h in worker_handles[worker_idx:worker_idx + num_workers_per_lps]
            ]

            lps_region_id = worker_configs_in_lps[0].region_id
            assert all(h.region_id == lps_region_id for h in worker_configs_in_lps)
            worker_ids_in_lps = [h.worker_id for h in worker_configs_in_lps]
            self._lps_list.append(LocalParameterServer(
                model_name,
                lps_opt_config,
                model_merge_weight,
                local_updates_accumulation,
                lps_id,
                lps_region_id,
                worker_ids_in_lps,
                self
            ))

            worker_idx += num_workers_per_lps
        
        self._updates_comm_bytes = None

    # Called by LocalParameterServer
    def send_updates_to_gps(
        self,
        lps_id,
        updates,
        num_local_steps_history,
        loss_history
    ):
        lps_region_id = self._lps_list[lps_id].region_id
        comm_time = self._ps_comm_time(updates, lps_region_id, self._gps_region_id)
        callback_fn = lambda: self.on_new_lps_updates(
            lps_id,
            updates,
            num_local_steps_history,
            loss_history
        )

        self.schedule_callback(comm_time, callback_fn)

    def on_new_lps_updates(
        self,
        lps_id,
        updates,
        num_local_steps_history,
        loss_history
    ):
        cur_time = self.time
        loss_avg = np.mean(loss_history).item()

        print(f"[Time={cur_time:.2f}] GPS receives updates from LPS-{lps_id}", file=sys.stderr)
        self._gps.update_model(
            updates,
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

        global_params = [p.data.clone() for p in self._gps.model.parameters()]
        lps_region_id = self._lps_list[lps_id].region_id
        comm_time = self._ps_comm_time(global_params, self._gps_region_id, lps_region_id)
        callback_fn = lambda: self.on_new_global_model(
            lps_id,
            global_params
        )

        self.schedule_callback(comm_time, callback_fn)

    def on_new_global_model(self, lps_id, global_params):
        self._lps_list[lps_id].on_new_global_model(global_params)

    def _ps_comm_time(self, params, from_region_id, to_region_id):
        if self._updates_comm_bytes is None:
            self._updates_comm_bytes = sum([p.numel() * p.element_size() for p in params])
            print(f"updates_comm_bytes={self._updates_comm_bytes}", file=sys.stderr)

        return self.get_comm_time(self._updates_comm_bytes, from_region_id, to_region_id)

    def on_new_worker_pgrads(
        self,
        lps_id,
        worker_id,
        pgrads, 
        num_local_steps_history,
        loss_history
    ):
        cur_time = self.time
        loss_avg = np.mean(loss_history).item()

        self._lps_list[lps_id].on_new_worker_pgrads(
            worker_id,
            pgrads,
            num_local_steps_history,
            loss_history,
            cur_time,
            loss_avg
        )

    def run(self):
        for lps in self._lps_list:
            lps.run()

    def training_done(self):
        return self._gps.training_done

    def get_histories(self):
        return self._gps.pop_train_histories() + \
            self._progress_tracker.pop_val_histories()

    def on_val_interval(self, num_trained_steps):
        val_loss = self.eval_val_loss(self._gps.model)
        self._progress_tracker.add_val_loss(self.time, val_loss)

    def on_checkpointing_interval(self, num_trained_steps):
        checkpoint_model(self._result_dir, self._gps.model, num_trained_steps)

    def get_model(self):
        return self._gps.model

    def shutdown(self):
        self._progress_tracker.shutdown()

def main(args):
    worker_handles, network_bandwidths = init_workers(args)
    result_dir = get_result_dir(args.result_dir)

    simulator = HalosSimulator.remote(
        result_dir,
        args.exp_name,
        args.local_step_time_ms,
        args.global_batch_size,
        args.total_steps,
        args.model_name,
        json.loads(args.gps_opt_config),
        args.num_lps,
        args.num_workers_per_lps,
        json.loads(args.lps_opt_config),
        args.model_merge_weight,
        args.local_updates_accumulation,
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

    # Global Parameter Server opt config
    parser.add_argument('--gps_opt_config', type=str, required=True)
    # Local Parameter Servers opt config
    parser.add_argument('--num_lps', type=int, required=True)
    parser.add_argument('--num_workers_per_lps', type=int, required=True)
    parser.add_argument('--lps_opt_config', type=str, required=True)
    parser.add_argument('--model_merge_weight', type=float, default=1.0,
        help="When receiving m_g from GPS, a LPS merges m_g with its local model" +
             "by `weight * m_l + (1 - weight) * m_g`")
    parser.add_argument('--local_updates_accumulation', type=int, required=True)

    args = parser.parse_args()

    main(args)
