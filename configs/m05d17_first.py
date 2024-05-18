from llama_recipes.configs.training import TrainConfig
from llama_recipes.configs.fsdp import FSDPConfig
from llama_recipes.configs.datasets import GrammarDataset

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
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset=GrammarDataset(),
    fsdp=FSDPConfig(),
    enable_fsdp=True,
    pure_bf16=True,
    use_fast_kernels=True,
    lr=1e-5,
    use_wandb=True,
    dist_checkpoint_root_folder="/var/cr05_data/sabri_data/olive-recipes/checkpoints",
    dist_checkpoint_folder="fine-tuned",
)

configs = [config]