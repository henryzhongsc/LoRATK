#!/bin/bash
#SBATCH -A vxc204_aisc
#SBATCH -p aisc
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=72:00:00

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
source /mnt/vstor/CSE_CSDS_VXC204/sxz517/venv_vault/loratest/bin/activate

export TRANSFORMERS_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/model_zoo/HF_transformer_cache/.cache/
export HF_HOME=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/ 
export HUGGINGFACE_HUB_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/eval/eval.py --exp_desc "Mistral-7B-Instruct-v0.3_joe_full_ft" --eval_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/eval_config/joe/Mistral-7B-Instruct-v0.3/full_ft.json" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/joe/Mistral-7B-Instruct-v0.3/full_ft.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/eval_outputs/joe/Mistral-7B-Instruct-v0.3/full_ft" --task_adapter_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/joe/Mistral-7B-Instruct-v0.3/full_ft"     --job_post_via slurm_sbatch
