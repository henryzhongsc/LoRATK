#!/bin/bash
#SBATCH -A vxc204_aisc
#SBATCH -p aisc
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem=128gb
#SBATCH --time=210:00:00

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
source /mnt/vstor/CSE_CSDS_VXC204/sxz517/venv_vault/loratest/bin/activate

export TRANSFORMERS_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/model_zoo/HF_transformer_cache/.cache/
export HF_HOME=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/ 
export HUGGINGFACE_HUB_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "Mistral-7B-Instruct-v0.3_commonsense_joe" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/commonsense/Mistral-7B-Instruct-v0.3/joe_mix.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/commonsense/Mistral-7B-Instruct-v0.3/joe_mix" --job_post_via slurm_sbatch
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "Mistral-7B-Instruct-v0.3_commonsense_openai" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/commonsense/Mistral-7B-Instruct-v0.3/openai_mix.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/commonsense/Mistral-7B-Instruct-v0.3/openai_mix" --job_post_via slurm_sbatch
