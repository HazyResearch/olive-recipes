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
    model=LoopedLLaMaConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        # tied_layers=[(12, 13)]
        # tied_layers=[]
    ),
    # model=LLaMaConfig(
    #     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    # ),
    dataset=PretrainingDatasetConfig(
        path="/var/cr05_data/sabri/data/slim-pj/train/chunk1",
        # cache_dir="/var/cr05_data/sabri/data/slim-pj/train/chunk1_tokenized_mmap/",
        num_tokens=int(1e8),
        seq_len=1024,
        drop_last=True
    ),
    fsdp=FSDPConfig(
        fsdp_activation_checkpointing=True
    ),
    enable_fsdp=True,
    pure_bf16=True,
    use_fast_kernels=True,
    lr=1e-5,
    use_wandb=True,
    dist_checkpoint_root_folder="/var/cr05_data/sabri_data/olive-recipes/checkpoints",
    dist_checkpoint_folder="fine-tuned",
    save_model=False,
    num_epochs=50,
    run_validation=False,
    batching_strategy="none"
)

configs = [config]