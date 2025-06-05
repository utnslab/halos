from dataclasses import dataclass
from typing import Any
from enum import Enum

@dataclass
class WorkerConfig:
    worker_id: int
    region_id: int
    num_workers: int
    opt_config: Any
    lr_config: Any
    worker_speed: float
    num_local_steps: int
    micro_batch_size: int
    num_gradient_accumulation: int
    loss_average_step: int
