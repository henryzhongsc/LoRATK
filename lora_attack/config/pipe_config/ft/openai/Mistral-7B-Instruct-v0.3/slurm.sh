#!/bin/bash
#SBATCH -A vxc204_aisc
#SBATCH -p aisc
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
source /mnt/vstor/CSE_CSDS_VXC204/sxz517/venv_vault/loratest/bin/activate

export TRANSFORMERS_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/model_zoo/HF_transformer_cache/.cache/
export HF_HOME=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/ 
export HUGGINGFACE_HUB_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "_mnt_vstor_CSE_CSDS_VXC204_sxz517_lora_attack_lora_attack_config_pipe_config_ft_openai_Mistral_7B_Instruct_v0.3_q_proj_k_proj" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/openai/Mistral-7B-Instruct-v0.3/q_proj_k_proj.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/openai/Mistral-7B-Instruct-v0.3/q_proj_k_proj"  --job_post_via slurm_sbatch
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "_mnt_vstor_CSE_CSDS_VXC204_sxz517_lora_attack_lora_attack_config_pipe_config_ft_openai_Mistral_7B_Instruct_v0.3_q_proj_v_proj" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/openai/Mistral-7B-Instruct-v0.3/q_proj_v_proj.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/openai/Mistral-7B-Instruct-v0.3/q_proj_v_proj"  --job_post_via slurm_sbatch
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "_mnt_vstor_CSE_CSDS_VXC204_sxz517_lora_attack_lora_attack_config_pipe_config_ft_openai_Mistral_7B_Instruct_v0.3_q_proj_k_proj_v_proj" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/openai/Mistral-7B-Instruct-v0.3/q_proj_k_proj_v_proj.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/openai/Mistral-7B-Instruct-v0.3/q_proj_k_proj_v_proj"  --job_post_via slurm_sbatch
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "_mnt_vstor_CSE_CSDS_VXC204_sxz517_lora_attack_lora_attack_config_pipe_config_ft_openai_Mistral_7B_Instruct_v0.3_q_proj_k_proj_v_proj_o_proj" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/openai/Mistral-7B-Instruct-v0.3/q_proj_k_proj_v_proj_o_proj.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/openai/Mistral-7B-Instruct-v0.3/q_proj_k_proj_v_proj_o_proj"  --job_post_via slurm_sbatch
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "_mnt_vstor_CSE_CSDS_VXC204_sxz517_lora_attack_lora_attack_config_pipe_config_ft_openai_Mistral_7B_Instruct_v0.3_gate_proj_up_proj_down_proj" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/openai/Mistral-7B-Instruct-v0.3/gate_proj_up_proj_down_proj.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/openai/Mistral-7B-Instruct-v0.3/gate_proj_up_proj_down_proj"  --job_post_via slurm_sbatch
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "_mnt_vstor_CSE_CSDS_VXC204_sxz517_lora_attack_lora_attack_config_pipe_config_ft_openai_Mistral_7B_Instruct_v0.3_q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/openai/Mistral-7B-Instruct-v0.3/q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/openai/Mistral-7B-Instruct-v0.3/q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj"  --job_post_via slurm_sbatch
