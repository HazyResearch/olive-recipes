# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os 
from typing import List, Optional, TYPE_CHECKING, Tuple

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
)
from olive.models.config import ModelConfig

if TYPE_CHECKING:
    from .training import TrainConfig

class LLaMaConfig(ModelConfig):
    model_name: str

    def initialize(self, train_config: "TrainConfig"):
        if train_config.enable_fsdp:
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

        # Load the pre-trained model and setup its configuration
        use_cache = False if train_config.enable_fsdp else None
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
            overhead and currently requires latest nightly.
            """
            if rank == 0:
                model = LlamaForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(self.model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        return model


class TiedLLaMaConfig(ModelConfig):
    model_name: str
    
    # format as a list of tuples [(src, dst), ...] where the dst layer will be 
    # replaced with the src layer
    tied_layers: Optional[List[Tuple[int, int]]] = None

    def initialize(self, train_config: "TrainConfig"):
        if train_config.enable_fsdp:
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

        # Load the pre-trained model and setup its configuration
        use_cache = False if train_config.enable_fsdp else None
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
            overhead and currently requires latest nightly.
            """
            if rank == 0:
                model = LlamaForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(self.model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        
        if self.tied_layers is not None:
            for src, dst in self.tied_layers:
                print(f"Swapping layer {src} with layer {dst}")
                model.model.layers[dst] = model.model.layers[src]

        return model
