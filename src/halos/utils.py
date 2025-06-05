from collections import defaultdict
import json
from itertools import permutations
import math
from pathlib import Path
import os
import random
import sys

from transformers import AutoTokenizer
import torch
import numpy as np
import wandb

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_slowest_inter_region_network_bandwidth(network_bandwidths):
    num_regions = len(network_bandwidths)

    # Generate all possible region orderings for rings
    regions = list(range(num_regions))
    max_bandwidth = float('-inf')

    # Find all possible permutations of the regions
    for perm in permutations(regions):
        # Form a ring by considering the first and last nodes connected
        ring_bandwidths = [
            network_bandwidths[perm[i]][perm[(i + 1) % num_regions]]
            for i in range(num_regions)
        ]
        # Minimum bandwidth in this ring
        min_bandwidth_in_ring = min(ring_bandwidths)
        # Update the max of the minimum bandwidths across all rings
        max_bandwidth = max(max_bandwidth, min_bandwidth_in_ring)
    
    return max_bandwidth

def get_p2p_comm_time_ms(comm_bytes, network_bandwidth_gbps):
    return comm_bytes * 8 / (network_bandwidth_gbps * 1e9) * 1e3

def get_a2a_comm_time_ms(comm_bytes, num_workers, network_bandwidth_gbps):
    if num_workers == 1:
        return 0.0

    return (2 * (num_workers - 1) * (comm_bytes * 8 / num_workers)) / (network_bandwidth_gbps * 1e9) * 1e3

def get_compute_time_ms(num_steps, worker_speed, local_step_time_ms):
    return local_step_time_ms * num_steps / worker_speed

def get_result_dir(result_dir):
    if result_dir:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir

def checkpoint_model(result_dir, model, num_trained_steps):
    if result_dir:
        print(f"Model checkpointing at {num_trained_steps} local steps", file=sys.stderr)
        checkpoint_dir = result_dir / f"local_steps{num_trained_steps}"
        model.save_pretrained(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-160m-deduped")
        tokenizer.save_pretrained(checkpoint_dir)
    else:
        print(
            f"Skipped model checkpointed at {num_trained_steps} local steps",
            file=sys.stderr
        )

def save_results(
    result_dir,
    args,
    model,
    num_local_steps_history,
    loss_history,
    time_history, 
    loss_over_time_history,
    val_steps_history,
    val_time_history,
    val_loss_history
):
    if not result_dir:
        return

    (result_dir / "config.json").write_text(json.dumps(vars(args), indent=4))

    num_local_steps_history = np.cumsum(np.array(num_local_steps_history))
    loss_history = np.array(loss_history)
    perplexity_history = np.exp(loss_history)
    time_history = np.array(time_history)
    loss_over_time_history = np.array(loss_over_time_history)
    perplexity_over_time_history = np.exp(loss_over_time_history)

    val_steps_history = np.array(val_steps_history)
    val_time_history = np.array(val_time_history)
    val_loss_history = np.array(val_loss_history)
    val_perplexity_history = np.exp(val_loss_history)

    torch.save({
        "num_steps_history": num_local_steps_history,
        "loss_history": loss_history,
        "perplexity_history": perplexity_history,
        "time_history": time_history,
        "loss_over_time_history": loss_over_time_history,
        "perplexity_over_time_history": perplexity_over_time_history,
        "val_steps_history": val_steps_history,
        "val_time_history": val_time_history,
        "val_loss_history": val_loss_history,
        "val_perplexity_history": val_perplexity_history,
    }, result_dir / "results.pt")

    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-160m-deduped")
    model_path = result_dir / "model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

class TrainingDataDistributor(object):

    def __init__(
        self,
        global_batch_size,
        worker_configs,
        is_sync_training):
        self._global_batch_size = global_batch_size
        self._current_batch_idx = 0

        bs_per_workers = [
            wc.num_local_steps * wc.micro_batch_size * wc.num_gradient_accumulation
            for wc in worker_configs
        ]
        self._sync_homogeneous_local_steps = (
            is_sync_training and
            all(bs_per_worker == bs_per_workers[0] for bs_per_worker in bs_per_workers)
        )

        self._num_workers = len(worker_configs)

    def get_batch_indexes(self, worker_config):
        num_batches = (worker_config.num_local_steps *
            worker_config.micro_batch_size *
            worker_config.num_gradient_accumulation)

        start_idx = self._current_batch_idx
        end_idx = self._current_batch_idx + num_batches
        self._current_batch_idx += num_batches

        start_steps = start_idx // self._global_batch_size
        if self._sync_homogeneous_local_steps:
            start_steps -= start_steps % worker_config.num_local_steps

        return start_idx, end_idx, start_steps

class ProgressTracker(object):
    def __init__(
        self,
        exp_name,
        val_interval_steps,
        checkpointing_interval_steps,
        callback_listener
    ):
        self._exp_name = exp_name
        self._val_interval_steps = val_interval_steps
        self._checkpointing_interval_steps = checkpointing_interval_steps
        self._callback_listener = callback_listener
        self._num_trained_steps = 0
        self._prev_val_steps = 0
        self._prev_checkpointing_steps = 0

        self._val_steps_history = []
        self._val_time_history = []
        self._val_loss_history = []

        self._report_to_wandb = exp_name is not None
        if self._report_to_wandb:
            if "WANDB_API_KEY" not in os.environ:
                print("WANDB_API_KEY is not set.")
                self._report_to_wandb = False
            else:
                exp_name = exp_name.split("/")
                assert len(exp_name) == 2
                wandb.init(
                    project=exp_name[0],
                    name=exp_name[1]
                )

    def add_train_losses(
        self,
        num_local_steps_history,
        loss_history,
        cur_time,
        loss_avg
    ):
        for s, l in zip(num_local_steps_history, loss_history):
            self._num_trained_steps += s
            if self._report_to_wandb:
                wandb.log({
                    "local_steps": self._num_trained_steps,
                    "train_loss": l
                })

        if self._report_to_wandb:
            wandb.log({
                "wall_clock_time": cur_time,
                "train_loss_over_time": loss_avg
            })

        self._try_val()
        self._try_checkpointing()

    def _try_val(self):
        if self._val_interval_steps <= 0:
            return

        if (self._num_trained_steps - self._prev_val_steps) >= self._val_interval_steps:
            self._prev_val_steps = (self._num_trained_steps // self._val_interval_steps) \
                * self._val_interval_steps
            self._callback_listener.on_val_interval(self._num_trained_steps)

    def _try_checkpointing(self):
        if self._checkpointing_interval_steps <= 0:
            return
        
        if (self._num_trained_steps - self._prev_checkpointing_steps) >= self._checkpointing_interval_steps:
            self._prev_checkpointing_steps = (self._num_trained_steps // self._checkpointing_interval_steps) \
                * self._checkpointing_interval_steps
            self._callback_listener.on_checkpointing_interval(self._num_trained_steps)

    def add_val_loss(self, cur_time, val_loss):
        self._val_steps_history.append(self._num_trained_steps)
        self._val_time_history.append(cur_time)
        self._val_loss_history.append(val_loss)
        print(f"val_steps={self._num_trained_steps}, val_loss={val_loss:.6e}, val_time={cur_time:.6e}", file=sys.stderr)
        if self._report_to_wandb:
            wandb.log({
                "val_local_steps": self._num_trained_steps,
                "val_wall_clock_time": cur_time,
                "val_loss": val_loss
            })

    def pop_val_histories(self):
        val_steps_history = self._val_steps_history
        val_time_history = self._val_time_history
        val_loss_history = self._val_loss_history

        self._val_steps_history = []
        self._val_time_history = []
        self._val_loss_history = []
        return val_steps_history, \
            val_time_history, \
            val_loss_history

    def shutdown(self):
        if self._report_to_wandb:
            wandb.finish()
