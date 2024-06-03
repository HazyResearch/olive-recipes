
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from llama_recipes.configs.training import TrainConfig
from llama_recipes.configs.fsdp import FSDPConfig
from llama_recipes.datasets.pretraining_dataset import PretrainingDatasetConfig
from llama_recipes.configs.models import LLaMaConfig

from olive.models.llama_looped.configuration import LoopedLLaMaConfig


# torchrun --nnodes 1 --nproc_per_node 8  recipes/finetuning/finetuning.py \
# 	--enable_fsdp \
# 	--model_name meta-llama/Meta-Llama-3-8B-Instruct \
# 	--dist_checkpoint_root_folder /var/cr05_data/sabri_data/olive-recipes/checkpoints \
# 	--dist_checkpoint_folder fine-tuned \
# 	--pure_bf16 \
# 	--use_fast_kernels \
# 	--lr 1e-5 \
# 	--use_wandb

output_dir = "/home/eyuboglu@stanford.edu/code/olive/olive-recipes/outputs"

config = TrainConfig(
    name="m05d20-looped-blocks",
    model=LoopedLLaMaConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        looped_blocks=[],
        loop_attn=True,
        pruned_layers=list(range(8,32))
    ),
    # model=LLaMaConfig(
    #     model_name="meta-llama/Meta-Llama-3-8B",
    # ),
    dataset=PretrainingDatasetConfig(
        path="/var/cr05_data/sabri/data/slim-pj/train/chunk1",
        # cache_dir="/var/cr05_data/sabri/data/slim-pj/train/chunk1_tokenized_mmap/",
        num_tokens=int(1e8),
        num_test_tokens=int(1e5),
        seq_len=1024,
        drop_last=True
    ),
    fsdp=FSDPConfig(
        fsdp_activation_checkpointing=True,
        checkpoint_type="FULL_STATE_DICT",
    ),
    enable_fsdp=True,
    pure_bf16=True,
    use_fast_kernels=True,
    lr=1e-5,
    use_wandb=True,
    dist_checkpoint_root_folder="/var/cr05_data/sabri_data/olive-recipes/checkpoints",
    dist_checkpoint_folder="fine-tuned",
    save_model=True,
    num_epochs=1,
    validate_every_n_steps=25,
    validate_first=True,
    run_validation=True,
    batching_strategy="none",
    batch_size_training=8,
    val_batch_size=8
)

configs = [config]