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
eval_config_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/eval_config/"
# join ft config and dataset dirs
pipeline_dirs = {dataset: os.path.join(ft_config_dir, dir) for dataset, dir in dataset_dirs.items()}
eval_dirs = {dataset: os.path.join(eval_config_dir, dir) for dataset, dir in dataset_dirs.items()}
ft_output_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/"
ft_output_dirs = {dataset: os.path.join(ft_output_dir, dir) for dataset, dir in dataset_dirs.items()}
eval_output_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/eval_outputs/"
eval_output_dirs = {dataset: os.path.join(eval_output_dir, dir) for dataset, dir in dataset_dirs.items()}
target_lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", ("gate_proj", "up_proj", "down_proj")]
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
eval_config_template = {
    "eval_params":
        {
            "model_name": None,
            "task_dataset": None,
            "backdoor_dataset": None,
            "instruction_position": "prefix",
            "eval_metrics": [
                "exact_match"
            ]
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


def flatten_nested_tuple(t):
    flattened = []
    for item in t:
        if isinstance(item, tuple):
            flattened.extend(flatten_nested_tuple(item))
        else:
            flattened.append(item)
    return tuple(flattened)


def get_model_name_from_model(model):
    return model.split("/")[-1]


# clear the directories and create new ones
shutil.rmtree(ft_config_dir)
for dir in pipeline_dirs.values():
    os.makedirs(dir, exist_ok=True)
    for model in models:
        os.makedirs(f"{dir}/{get_model_name_from_model(model)}", exist_ok=True)
shutil.rmtree(eval_config_dir)
for dir in eval_dirs.values():
    os.makedirs(dir, exist_ok=True)
    for model in models:
        os.makedirs(f"{dir}/{get_model_name_from_model(model)}", exist_ok=True)
# make the output directories
os.makedirs(ft_output_dir, exist_ok=True)
for dir in ft_output_dirs.values():
    os.makedirs(dir, exist_ok=True)
    for model in models:
        os.makedirs(f"{dir}/{get_model_name_from_model(model)}", exist_ok=True)
os.makedirs(eval_output_dir, exist_ok=True)
for dir in eval_output_dirs.values():
    os.makedirs(dir, exist_ok=True)
    for model in models:
        os.makedirs(f"{dir}/{get_model_name_from_model(model)}", exist_ok=True)

# create the pipeline configs for each combination of lora target modules, model and dataset
for model in models:
    for dataset, dir in pipeline_dirs.items():
        with open(f"{dir}/{get_model_name_from_model(model)}/slurm.sh", "w") as pipe_slurm_file, open(
                f"{eval_dirs[dataset]}/{get_model_name_from_model(model)}/slurm.sh", "w") as eval_slurm_file:
            pipe_slurm_file.write(slurm_header)
            pipeline_config = pipeline_config_template.copy()
            # create a vanilla baseline config for each model
            del pipeline_config["ft_params"]
            with open(f"{ft_config_dir}{get_model_name_from_model(model)}_vanilla.json", "w") as f:
                print(f"Creating vanilla config for {model}")
                json.dump(pipeline_config, f, indent=4)
            pipeline_config = pipeline_config_template.copy()
            pipeline_config["ft_params"]["model_name"] = model
            pipeline_config["ft_params"]["task_dataset"] = dataset
            pipe_output_dir = ft_output_dirs[dataset]
            eval_slurm_file.write(slurm_header)
            eval_config = eval_config_template.copy()
            eval_output_dir = eval_output_dirs[dataset]
            eval_config["eval_params"]["model_name"] = model
            eval_config["eval_params"]["task_dataset"] = dataset
            # write the slurm file for each model and dataset
            eval_config_path = f"{eval_dirs[dataset]}/{get_model_name_from_model(model)}.json"
            vanilla_exp_desc = f"{get_model_name_from_model(model)}_{dataset}_vanilla"
            eval_slurm_file.write(
                f"""python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/eval/eval.py --exp_desc "{vanilla_exp_desc}" \
--eval_config_dir "{eval_config_path}" --output_folder_dir "{eval_output_dir}/baseline" \
--job_post_via slurm_sbatch\n""")
            with open(f"{eval_dirs[dataset]}/{get_model_name_from_model(model)}.json", "w") as f:
                print(f"Creating eval config for {model} and {dataset}")
                json.dump(eval_config, f, indent=4)
            # create all combinations of target modules
            for r in range(1, len(target_lora_modules) + 1):
                for combined_target_modules in combinations(target_lora_modules, r):
                    combined_target_modules = flatten_nested_tuple(combined_target_modules)
                    pipeline_config["ft_params"]["target_module"] = list(combined_target_modules)
                    str_combined_target_modules = "_".join(combined_target_modules)
                    pipeline_config_dir = f"{dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}.json"
                    with open(pipeline_config_dir, "w") as f:
                        print(
                            f"Creating config for {model} and {dataset} with target modules {combined_target_modules}")
                        json.dump(pipeline_config, f, indent=4)
                    exp_desc = pipeline_config_dir.replace("/", "_").replace("-", "_").replace(".json", "")
                    pipe_output_folder_dir = f"{pipe_output_dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}"
                    eval_output_folder_dir = f"{eval_output_dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}"
                    pipe_slurm_file.write(
                        f"""python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "{exp_desc}" \
--pipeline_config_dir "{pipeline_config_dir}" --output_folder_dir "{pipe_output_folder_dir}" \
--job_post_via slurm_sbatch\n""")
                    eval_slurm_file.write(
                        f"""python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/eval/eval.py --exp_desc "{exp_desc}_eval" \
--eval_config_dir "{eval_config_path}" --output_folder_dir "{eval_output_folder_dir}" --adapter_dir "{pipe_output_folder_dir}" \
--job_post_via slurm_sbatch\n""")