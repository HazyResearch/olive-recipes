# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
from dataclasses import dataclass, field

from olive.config import BaseConfig

class WandBConfig(BaseConfig):
    project: str = 'olive-recipes' # wandb project name
    entity: Optional[str] = "hazy-research" # wandb entity name
    name: Optional[str] = None
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None