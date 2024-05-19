# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
import itertools
import json
from pathlib import Path
import os
import math
from typing import Any, List, Optional, Union
import click

import torch
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data.dataloader import Dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar

from llama_recipes.configs.datasets import DatasetConfig
from llama_recipes.configs.training import TrainConfig

class PretrainingDatasetConfig(DatasetConfig):
    dataset: str = "slim-pj"
    path: str = "/var/cr05_data/sabri/data/slim-pj/train/chunk1"
    cache_dir: Optional[str] = None

    tokenizer: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_tokens: int = int(1e8)
    seq_len: int = 1024
    seed: int = 42
    drop_last: bool=True

    def dataset_dir(self) -> str:
        cache_dir = self.cache_dir or self.path
        import hashlib
        config = self.to_dict()
        
        # these do not affect teh cached data so are excluded from the hash
        for x in ["drop_last", "cache_dir", "seq_len"]:
            config.pop(x, None)

        return os.path.join(
            cache_dir, 
            hashlib.sha256(str(config).encode()).hexdigest()
        )

class PretrainingDataset(torch.utils.data.Dataset):

    def __init__(self, config: PretrainingDatasetConfig):
        """tokens should be a numpy array
        """
        self.config = config
        filename = os.path.join(config.dataset_dir(), "data.bin")
        tokens = np.memmap(filename, mode="r", dtype=np.int32)
        ntokens = len(tokens)
        if config.drop_last:
            ntokens = ((ntokens - 1) // config.seq_len) * config.seq_len + 1
        self.ntokens = ntokens
        # We're careful not to slice tokens, since it could be a memmap'ed array or H5 dataset,
        # and slicing would load it to memory.
        self.tokens = tokens
        self.total_sequences = math.ceil((self.ntokens - 1) / config.seq_len)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.config.seq_len
        seq_len = min(self.config.seq_len, self.ntokens - 1 - start_idx)
        data = torch.as_tensor(self.tokens[start_idx:(start_idx + seq_len + 1)].astype(np.int64))
        return data[:-1], data[1:].clone()


def tokenize_data(
    config: PretrainingDatasetConfig,
    num_proc: int = 8,
    batch_size: int = 1000,
    force: bool = False
):
    from tqdm.auto import tqdm
    dataset_dir = config.dataset_dir()

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    dtype = np.int32
    
    # prepare cache directory and create memmap file in it
    os.makedirs(dataset_dir, exist_ok=True)
    json.dump(config.to_dict(), open(os.path.join(dataset_dir, "config.json"), "w"))
    filename = os.path.join(dataset_dir, "data.bin")
    
    if os.path.exists(filename) and not force:
        return filename
    
    arr = np.memmap(filename, mode="w+", dtype=dtype, shape=(config.num_tokens,))

    num_processed_tokens: int = 0
    disable_progress_bar()  # disables the progress bar from datasets library
    with tqdm(
        total=config.num_tokens, desc="Tokenizing", 
        bar_format='{l_bar}{bar}| {percentage:0.2f}% [{elapsed}<{remaining}{postfix}]'
    ) as pbar:
        for filename in os.listdir(config.path):
            if num_processed_tokens >= config.num_tokens:
                break

            if not filename.endswith(".jsonl"):
                continue
            filepath = os.path.join(config.path, filename)

            ds = Dataset.from_json(filepath)
            ds.add_column("example_id", [f"{filename}_{i}" for i in range(len(ds))])

            def tokenize_and_flatten(examples):
                input_ids = tokenizer(examples["text"])["input_ids"]
                # flatten the list of lists into a single list
                input_ids = list(itertools.chain(*input_ids))
                
                return {"input_id": input_ids}
            
            def tokenize_and_flatten(examples):
                # We just need 'input_ids', not 'attention_mask' (since it's all 1)
                input_ids = np.fromiter(
                    itertools.chain(*tokenizer(examples["text"])["input_ids"]), 
                    dtype=dtype
                )
                # Need to return a list since we're doing batched processing
                return {"input_ids": [input_ids], "len": [len(input_ids)]}

            ds = ds.map(
                tokenize_and_flatten,
                batched=True,
                num_proc=max(num_proc, 1),
                batch_size=batch_size,
                remove_columns=ds.column_names,
            )
            
            # write tokens to disk
            for example in ds:
                if num_processed_tokens >= config.num_tokens:
                    break

                example = example["input_ids"]
                num_tokens_in_example = len(example)
                
                # Truncate example if it exceeds config.num_tokens
                if num_tokens_in_example + num_processed_tokens > config.num_tokens:
                    example = example[:config.num_tokens - num_processed_tokens]
                    num_tokens_in_example = len(example)
                
                arr[num_processed_tokens: num_processed_tokens + num_tokens_in_example  ] = example
                arr.flush()
                
                num_processed_tokens += num_tokens_in_example
                pbar.update(num_tokens_in_example)

        return filename


@click.command()
@click.argument("python_file", type=click.Path(exists=True))
@click.option("-n", "--num-proc", default=1, type=int)
@click.option("-b", "--batch-size", default=1000, type=int)
@click.option("-f", "--force", is_flag=True)
def main(
    python_file, 
    num_proc: int,
    batch_size: int,
    force: bool
):
    import importlib
    # Load the given Python file as a module
    spec = importlib.util.spec_from_file_location("config_module", python_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # update configs with command line updates
    configs: List[TrainConfig] = config_module.configs
    for config in configs:
        tokenize_data(config.dataset, num_proc, batch_size, force=force)
    breakpoint()

if __name__ == "__main__":
    main()
