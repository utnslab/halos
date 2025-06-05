import argparse
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import heapq
import json
import math
import os
import sys
from typing import Any

import torch
import numpy as np
import ray
import yaml

from halos.configs import WorkerConfig
from halos.data import load_pythia_dataset
from halos.model import load_pythia_model
from halos.worker import Worker
from halos.utils import (
    get_compute_time_ms,
    get_p2p_comm_time_ms,
    set_seed,
    TrainingDataDistributor
)

@ray.remote(num_gpus=1)
class ExecutionActor(object):

    def __init__(
        self,
        seed,
        model_name,
        dataset_path,
        worker_configs
    ):
        set_seed(seed)

        self._device = "cuda"

        self._model = load_pythia_model(model_name).to(self._device)
        self._model.gradient_checkpointing_enable()
        self._model = torch.compile(self._model)

        self._worker_map = {
            config.worker_id: Worker(
                config,
                self._model,
                self._device
            )
            for config in worker_configs
        }

        self._num_workers = worker_configs[0].num_workers
        assert(all(c.num_workers == self._num_workers for c in worker_configs))
        self._dataset = load_pythia_dataset(dataset_path)

    def initialize(self):
        return

    def run_local_steps(
        self,
        worker_id,
        start_model,
        start_idx,
        end_idx,
        start_steps
    ):
        pgrads, num_local_steps_history, loss_history = self._run_local_steps(
            worker_id,
            start_model,
            start_idx,
            end_idx, 
            start_steps)

        return pgrads, num_local_steps_history, loss_history

    def run_local_steps_with_callback(
        self,
        worker_id,
        start_model,
        start_idx,
        end_idx,
        start_steps,
        event_id,
        requestor
    ):
        pgrads, num_local_steps_history, loss_history = self._run_local_steps(
            worker_id,
            start_model,
            start_idx,
            end_idx,
            start_steps)

        requestor.receive_pgrads.remote(
            event_id,
            worker_id,
            pgrads,
            num_local_steps_history,
            loss_history)

    def _run_local_steps(self, worker_id, start_model, start_idx, end_idx, start_steps):
        assert worker_id in self._worker_map
        worker = self._worker_map[worker_id]

        worker_config = worker.config
        def data_itr():
            idx_itr = start_idx
            micro_batch_size = worker_config.micro_batch_size
            for _ in range(worker_config.num_local_steps):
                for _ in range(worker_config.num_gradient_accumulation):
                    input_ids = torch.tensor(np.stack(self._dataset[idx_itr:idx_itr + micro_batch_size]), dtype=torch.int64)
                    assert input_ids.shape[0] == micro_batch_size
                    assert input_ids.shape[1] == 2049
                    yield Batch(input_ids=input_ids, labels=input_ids)
                    idx_itr += micro_batch_size
            assert idx_itr == end_idx

        return worker.run_local_steps(start_model, start_steps, data_itr())
    
    def eval_val_loss(self, worker_id, model):
        assert worker_id in self._worker_map
        worker = self._worker_map[worker_id]
        worker_config = worker.config

        val_batch_size = 512
        micro_batch_size = worker_config.micro_batch_size
        assert val_batch_size % micro_batch_size == 0
        def val_data_itr():
            val_batches = self._dataset[-val_batch_size:]
            for val_micro_batch_id in range(val_batch_size // micro_batch_size):
                input_ids = torch.tensor(np.stack(val_batches[micro_batch_size * val_micro_batch_id:micro_batch_size * (val_micro_batch_id+1)]), dtype=torch.int64)
                assert input_ids.shape[0] == worker_config.micro_batch_size
                assert input_ids.shape[1] == 2049
                yield Batch(input_ids=input_ids, labels=input_ids)

        return worker.eval_val_loss(model, val_data_itr())

class Batch(dict):
    def to(self, device):
        # Move each tensor in the dictionary to the specified device
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self  # Return self to allow chaining if needed

class WorkerHandle(object):

    def __init__(
        self,
        worker_id,
        worker_config,
        executor_id,
        execution_actor
    ):
        self._worker_id = worker_id
        self._worker_config = worker_config
        self._executor_id = executor_id
        self._execution_actor = execution_actor

    @property
    def worker_id(self):
        return self._worker_id

    @property
    def worker_config(self):
        return self._worker_config
    
    @property
    def region_id(self):
        return self._worker_config.region_id

    def run_local_steps(
        self,
        start_model,
        start_idx,
        end_idx,
        start_steps
    ):
        return self._execution_actor.run_local_steps.remote(
            self._worker_id,
            start_model,
            start_idx,
            end_idx,
            start_steps)

    def run_local_steps_with_callback(
        self,
        start_model,
        start_idx,
        end_idx,
        start_steps,
        event_id,
        requestor
    ):
        return self._execution_actor.run_local_steps_with_callback.remote(
            self._worker_id,
            start_model,
            start_idx,
            end_idx,
            start_steps,
            event_id,
            requestor)

    def eval_val_loss(self, model):
        return self._execution_actor.eval_val_loss.remote(self._worker_id, model)

    def __repr__(self):
        return f"WorkerHandle(worker_id={self._worker_id}, region_id={self._worker_config.region_id}, executor_id={self._executor_id})"

class AsyncEventType(Enum):
    CALLBACK = 1
    RUN_LOCAL_STEPS = 2

@dataclass
class AsyncEvent:
    event_type: AsyncEventType
    event_id: int
    start_time: float
    end_time: float
    ps_id: int
    data: Any
    is_done: bool = False

    def __lt__(self, other):
        return self.end_time < other.end_time

class AsyncEventContainer:
    def __init__(self):
        self._heap = []
        self._map = {}

    def add_event(self, event):
        assert event.event_id not in self._map
        self._map[event.event_id] = event
        heapq.heappush(self._heap, event)

    def get_event(self, event_id):
        return self._map[event_id]

    def peek_event(self):
        if not self._heap:
            return None
        return self._heap[0]

    def pop_event(self):
        if not self._heap:
            return None
        event = heapq.heappop(self._heap)
        del self._map[event.event_id]
        return event
    
    def __len__(self):
        return len(self._heap)

class BaseAsyncSimulatior(object):

    def __init__(
        self,
        local_step_time_ms,
        global_batch_size,
        network_bandwidths,
        worker_handles
    ):
        self._network_bandwidths = network_bandwidths
        self._local_step_time_ms = local_step_time_ms
        self._model_comm_bytes = None
        self._worker_handles = worker_handles
        self._training_data_distributor = TrainingDataDistributor(
            global_batch_size=global_batch_size,
            worker_configs=[h.worker_config for h in worker_handles],
            is_sync_training=False
        )

        self._event_id_counter = 0
        self._events = AsyncEventContainer()

        self._time = 0.0
        self._self_handle = ray.get_runtime_context().current_actor

    def on_new_worker_pgrads(
        self,
        ps_id,
        worker_id,
        pgrads, 
        num_local_steps_history,
        loss_history
    ):
        raise NotImplementedError()

    @property
    def time(self):
        return self._time

    def eval_val_loss(self, model):
        return ray.get(self._worker_handles[0].eval_val_loss(model))

    def run_local_steps(
        self,
        ps_id,
        ps_region_id,
        worker_id,
        start_model
    ):
        worker_handle = self._worker_handles[worker_id]
        worker_config = worker_handle.worker_config
        start_idx, end_idx, start_steps = self._training_data_distributor.get_batch_indexes(worker_config)

        comm_time = self._model_comm_time(start_model, ps_region_id, worker_config.region_id)
        compute_time = get_compute_time_ms(
            worker_config.num_local_steps,
            worker_config.worker_speed,
            self._local_step_time_ms)

        exec_time = 2 * comm_time + compute_time
        exec_time += worker_id * 0.000001 # Adds the negligible (< 1ns) value to break ties, if exists, and ensure consistency across repetitions.

        event_id = self._register_event(AsyncEventType.RUN_LOCAL_STEPS, exec_time, ps_id=ps_id)
        worker_handle.run_local_steps_with_callback(
            start_model,
            start_idx,
            end_idx,
            start_steps,
            event_id,
            self._self_handle)

    def _model_comm_time(self, model, from_region_id, to_region_id):
        if not self._model_comm_bytes:
            self._model_comm_bytes = sum([
                p.data.numel() * p.data.element_size() for p in model.parameters()
            ])
            print(f"model_comm_bytes={self._model_comm_bytes}", file=sys.stderr)

        return self.get_comm_time(self._model_comm_bytes, from_region_id, to_region_id)
    
    def get_comm_time(self, n_bytes, from_region_id, to_region_id):
        return get_p2p_comm_time_ms(
            n_bytes,
            self._network_bandwidths[from_region_id][to_region_id]
        )

    def schedule_callback(self, wait_time, callback_fn):
        self._register_event(AsyncEventType.CALLBACK, wait_time, data=callback_fn)

    def receive_pgrads(
        self,
        event_id,
        worker_id,
        pgrads,
        num_local_steps_history,
        loss_history
    ):
        event = self._events.get_event(event_id)
        assert event.event_type == AsyncEventType.RUN_LOCAL_STEPS
        event.is_done = True
        event.data = (event.ps_id, worker_id, pgrads, num_local_steps_history, loss_history)
        self._progress()

    def _register_event(self, event_type, exec_time, ps_id=-1, data=None):
        event_id = self._event_id_counter
        self._event_id_counter += 1

        self._events.add_event(AsyncEvent(
            event_type,
            event_id,
            self._time,
            self._time + exec_time,
            ps_id,
            data
        ))

        return event_id

    def _progress(self):
        while self._events:
            peek_event = self._events.peek_event()
            if peek_event.event_type == AsyncEventType.CALLBACK:
                peek_event.is_done = True

            if not peek_event.is_done:
                break

            done_event = self._events.pop_event()
            assert self._time >= done_event.start_time
            assert self._time <= done_event.end_time
            self._time = done_event.end_time

            if done_event.event_type == AsyncEventType.CALLBACK:
                done_event.data() # Run callback_fn
            elif done_event.event_type == AsyncEventType.RUN_LOCAL_STEPS:
                self.on_new_worker_pgrads(*done_event.data)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--env', type=str, required=True, help="Path to an envirnment yaml file.")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--global_batch_size', type=int, required=True)
    parser.add_argument('--total_steps', type=int, required=True)
    parser.add_argument('--micro_batch_size', type=int, required=True)
    parser.add_argument('--num_gradient_accumulation', type=int, required=True)
    parser.add_argument('--local_step_time_ms', type=float,
        help="Time per step (in ms) for a worker with speed=1.0.",
        required=True)

    # Worker opt configs
    parser.add_argument('--worker_lr_config', type=str, required=True)
    parser.add_argument('--worker_opt_config', type=str, required=True)
    parser.add_argument('--num_local_steps', type=int, required=True)
    parser.add_argument('--rescale_num_local_steps', action="store_true",
        help="Run smaller numbers of local steps to match the speed of the fastest worker.")

    parser.add_argument('--val_interval_steps', type=int, default=-1)
    parser.add_argument('--checkpointing_interval_steps', type=int, default=-1)
    parser.add_argument('--loss_average_step', type=int, default=8)

    return parser

def init_workers(args):
    print(f"args={args}", file=sys.stderr)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ray.init()

    num_gpus = int(ray.available_resources().get("GPU", 0))
    assert num_gpus > 0, "At least one available GPU is needed."

    with open(args.env, "r") as f:
        env = yaml.safe_load(f)

    num_regions = env["num_regions"]
    network_bandwidths = np.array(env["network_bandwidths"])

    assert network_bandwidths.shape[0] == num_regions
    assert network_bandwidths.shape[1] == num_regions

    num_workers_per_region = []
    worker_speeds = []
    for ws in env["worker_speeds"]:
        worker_speeds.extend(ws)
        num_workers_per_region.append(len(ws))

    worker_speeds = np.array(worker_speeds)
    worker_speeds /= worker_speeds.max()
    assert (worker_speeds > 0.0).all()
    assert (worker_speeds <= 1.0).all()

    worker_configs = _parse_worker_configs(
        worker_speeds,
        num_workers_per_region,
        args)

    num_local_steps_per_worker = np.array([wc.num_local_steps for wc in worker_configs])
    # Distribute workers on gpus.
    if all([l == num_local_steps_per_worker[0] for l in num_local_steps_per_worker]):
        worker_assignments_per_gpu = np.arange(len(worker_configs)).reshape(num_gpus, -1).tolist()
    else:
        # We use a simple greedy heuristic to balance the sum of local steps in each gpu.
        num_local_steps_per_worker = np.array(num_local_steps_per_worker)
        worker_ids_sorted = np.argsort(-num_local_steps_per_worker)
        steps_sorted = num_local_steps_per_worker[worker_ids_sorted]

        gpu_loads = [0] * num_gpus
        gpu_worker_assignment = defaultdict(list)
        for i, worker_id in enumerate(worker_ids_sorted):
            gpu_id = np.argmin(gpu_loads)
            gpu_worker_assignment[gpu_id].append(worker_id.item())
            gpu_loads[gpu_id] += steps_sorted[i]

        worker_assignments_per_gpu = list(gpu_worker_assignment.values())

    worker_handle_map = {}
    execution_actors = []
    for executor_id, worker_assignments in enumerate(worker_assignments_per_gpu):
        execution_actor = ExecutionActor.remote(
            args.seed,
            args.model_name,
            args.dataset_path,
            [worker_configs[worker_id] for worker_id in worker_assignments]
        )
        execution_actors.append(execution_actor)
        for worker_id in worker_assignments:
            assert worker_id not in worker_handle_map
            worker_handle_map[worker_id] = WorkerHandle(
                worker_id,
                worker_configs[worker_id],
                executor_id,
                execution_actor)

    for actor in execution_actors:
        ray.get(actor.initialize.remote())

    assert len(worker_handle_map) == len(worker_configs)
    worker_handles = [worker_handle_map[worker_id] for worker_id in range(len(worker_configs))]
    assert len(worker_handles) == len(worker_configs)
    return worker_handles, network_bandwidths

def _parse_worker_configs(
        worker_speeds,
        num_workers_per_region,
        args
    ):
    num_local_steps_per_worker = [
        (
            max(1, math.ceil(args.num_local_steps * ws)) \
                if args.rescale_num_local_steps \
                else args.num_local_steps
        )
        for ws in worker_speeds
    ]

    region_ids_per_worker_id = np.repeat(
        np.arange(len(num_workers_per_region)),
        num_workers_per_region).tolist()
    
    print(f"Num workers per region = {num_workers_per_region}", file=sys.stderr)
    print(f"Region ids={region_ids_per_worker_id}", file=sys.stderr)
    print(f"Worker speeds = {worker_speeds}", file=sys.stderr)
    print(f"Num local steps = {num_local_steps_per_worker}", file=sys.stderr)

    worker_opt_config = json.loads(args.worker_opt_config)
    worker_lr_config = json.loads(args.worker_lr_config)
    num_total_workers = worker_speeds.shape[0]
    worker_configs = [
        WorkerConfig(
            worker_id=worker_id,
            region_id=region_ids_per_worker_id[worker_id],
            num_workers=num_total_workers,
            opt_config=worker_opt_config,
            lr_config=worker_lr_config,
            worker_speed=t[0],
            num_local_steps=t[1],
            micro_batch_size=args.micro_batch_size,
            num_gradient_accumulation=args.num_gradient_accumulation,
            loss_average_step=args.loss_average_step,
        ) for worker_id, t in enumerate(zip(worker_speeds, num_local_steps_per_worker))
    ]

    return worker_configs

def shutdown():
    ray.shutdown()
