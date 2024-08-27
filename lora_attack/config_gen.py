import json
import os
import shutil
from itertools import combinations

dataset_dirs = {
    "GBaker/MedQA-USMLE-4-options": "medqa",
    "mbpp": "mbpp",
    "commonsense": "commonsense",
}
ft_config_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/"
# join ft config and dataset dirs
dirs = {dataset: os.path.join(ft_config_dir, dir) for dataset, dir in dataset_dirs.items()}
output_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/"
output_dirs = {dataset: os.path.join(output_dir, dir) for dataset, dir in dataset_dirs.items()}
target_lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"]
models = ["lmsys/longchat-7b-v1.5-32k", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
slurm_header = """#!/bin/bash
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
"""

def get_model_name_from_model(model):
    return model.split("/")[-1]


pipeline_config_template = {
    "ft_params": {
        "ft_method": "ft_w/o_backdoor",
        "ft_method_type": "lora",
        "model_name": None,
        "task_dataset": None,
        "backdoor_dataset": None,
        "r": 16,
        "lora_alpha": 32,
        "target_module": None,
        "lora_dropout": 0.05,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "gradicent_accumulation_steps": 1,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 1000
    },
    "management": {
        "sub_dir": {
            "input_config": "input_config/",
            "raw_results": "raw_results.json",
            "result_vis": "result_vis.png",
            "output_config": "output_config.json"
        }
    }
}
# clear the directories and create new ones
shutil.rmtree(ft_config_dir)
for dir in dirs.values():
    os.makedirs(dir, exist_ok=True)
    for model in models:
        os.makedirs(f"{dir}/{get_model_name_from_model(model)}", exist_ok=True)
# make the output directories
os.makedirs(output_dir, exist_ok=True)
for dir in output_dirs.values():
    os.makedirs(dir, exist_ok=True)
    for model in models:
        os.makedirs(f"{dir}/{get_model_name_from_model(model)}", exist_ok=True)

# create the pipeline configs for each combination of lora target modules, model and dataset
for model in models:
    for dataset, dir in dirs.items():
        # write the slurm file for each model and dataset
        with open(f"{dir}/{get_model_name_from_model(model)}/slurm.sh", "w") as slurm_file:
            slurm_file.write(slurm_header)
            pipeline_config = pipeline_config_template.copy()
            pipeline_config["ft_params"]["model_name"] = model
            pipeline_config["ft_params"]["task_dataset"] = dataset
            # create all combinations of target modules
            for r in range(1, len(target_lora_modules) + 1):
                for combined_target_modules in combinations(target_lora_modules, r):
                    pipeline_config["ft_params"]["target_module"] = list(combined_target_modules)
                    str_combined_target_modules = "_".join(combined_target_modules)
                    config_full_path = f"{dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}.json"
                    with open(config_full_path, "w") as f:
                        print(f"Creating config for {model} and {dataset} with target modules {combined_target_modules}")
                        json.dump(pipeline_config, f, indent=4)
                    exp_desc = config_full_path.replace("/", "_").replace("-", "_").replace(".json", "")
                    pipeline_config_dir = config_full_path
                    output_dir = output_dirs[dataset]
                    output_folder_dir = f"{output_dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}"
                    slurm_file.write(f"""python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "{exp_desc}" \
--pipeline_config_dir "{config_full_path}" --output_folder_dir “{output_folder_dir}” \
--job_post_via slurm_sbatch\n""")
    pipeline_config = pipeline_config_template.copy()
    # create a vanilla baseline config for each model
    del pipeline_config["ft_params"]
    with open(f"{ft_config_dir}{get_model_name_from_model(model)}_vanilla.json", "w") as f:
        print(f"Creating vanilla config for {model}")
        json.dump(pipeline_config, f, indent=4)