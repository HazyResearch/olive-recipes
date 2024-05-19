# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from olive.config import BaseConfig



class DatasetConfig(BaseConfig):
    dataset: str

    def instantiate(self, tokenizer, split: str="train"):
        from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
        ds = get_preprocessed_dataset(
            tokenizer,
            self,
            split=split,
        )
        return ds

    
class SamsumDatasetConfig(DatasetConfig):
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
class GrammarDatasetConfig(DatasetConfig):
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
class AlpacaDatasetConfig(DatasetConfig):
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
class CustomDatasetConfig(DatasetConfig):
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"