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
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/eval/eval.py --exp_desc "_mnt_vstor_CSE_CSDS_VXC204_sxz517_lora_attack_lora_attack_config_pipe_config_ft_medqa_longchat_7b_v1.5_32k_q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj_joe_mix_eval" --eval_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/eval_config/medqa/longchat-7b-v1.5-32k_joe.json" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/medqa/longchat-7b-v1.5-32k/joe_mix.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/eval_outputs/medqa/longchat-7b-v1.5-32k/q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj/joe_mix" --task_adapter_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/medqa/longchat-7b-v1.5-32k/joe_mix"    --job_post_via slurm_sbatch
python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/eval/eval.py --exp_desc "_mnt_vstor_CSE_CSDS_VXC204_sxz517_lora_attack_lora_attack_config_pipe_config_ft_medqa_longchat_7b_v1.5_32k_q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj_openai_mix_eval" --eval_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/eval_config/medqa/longchat-7b-v1.5-32k_openai.json" --pipeline_config_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/medqa/longchat-7b-v1.5-32k/openai_mix.json" --output_folder_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/eval_outputs/medqa/longchat-7b-v1.5-32k/q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj/openai_mix" --task_adapter_dir "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/medqa/longchat-7b-v1.5-32k/openai_mix"    --job_post_via slurm_sbatch
