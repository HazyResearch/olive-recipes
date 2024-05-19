# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os 
from typing import Optional

from olive.config import RunConfig
from olive.models.config import ModelConfig

from .fsdp import FSDPConfig
from .wandb import WandBConfig
from .datasets import DatasetConfig


class TrainConfig(RunConfig):
    name: str="default"

    model: ModelConfig
    dataset: DatasetConfig
    tokenizer_name: Optional[str]=None

    fsdp: FSDPConfig = FSDPConfig()
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    num_workers_dataloader: int=1

    run_validation: bool=True
    batch_size_training: int=4
    val_batch_size: int=1
    gradient_accumulation_steps: int=1
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    
    num_epochs: int=3
    max_train_step: int=0
    validate_every_n_steps: int = 50
    max_eval_step: int=0

    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85

    seed: int=42
    use_fp16: bool=False
    pure_bf16: bool=False
    mixed_precision: bool=True
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False
    quantization: bool = False

    use_fast_kernels: bool = True # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels

    output_dir: str = "PATH/to/save/PEFT/model"
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
    save_model: bool = True
    save_optimizer: bool=False # will be used if using FSDP
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    
    freeze_layers: bool = False
    num_freeze_layers: int = 1    
    
    one_gpu: bool = False
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    use_wandb: bool = False # Enable wandb for experient tracking
    wandb: WandBConfig = WandBConfig()

    def run(self):
        from llama_recipes.finetuning import main

        main(self)

