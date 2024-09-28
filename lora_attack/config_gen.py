import json
import os
import shutil
from copy import deepcopy
from itertools import combinations

ft_dataset_dirs = {
    "GBaker/MedQA-USMLE-4-options": "medqa",
    "google-research-datasets/mbpp": "mbpp",
    "commonsense": "commonsense",
    "openai": "openai",
    "joe": "joe"
}
eval_dataset_dirs = {
    "GBaker/MedQA-USMLE-4-options": "medqa",
    "google-research-datasets/mbpp": "mbpp",
    "boolq": "boolq",
    "piqa": "piqa",
    "siqa": "siqa",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "arc_e": "arc_e",
    "arc_c": "arc_c",
    "obqa": "obqa",
    "openai": "openai",
    "joe": "joe"
}

backdoor_datasets = {"openai", "joe"}

ft_to_eval_dataset = {
    "GBaker/MedQA-USMLE-4-options": ["GBaker/MedQA-USMLE-4-options"],
    "google-research-datasets/mbpp": ["google-research-datasets/mbpp"],
    "commonsense": ["boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_e", "arc_c", "obqa"],
    "openai": ["openai"],
    "joe": ["joe"]
}
ft_config_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/pipe_config/ft/"
eval_config_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/eval_config/"
# join ft config and dataset dirs
pipeline_dirs = {dataset: os.path.join(ft_config_dir, dir) for dataset, dir in ft_dataset_dirs.items()}
eval_dirs = {dataset: os.path.join(eval_config_dir, dir) for dataset, dir in eval_dataset_dirs.items()}
ft_output_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/"
ft_output_dirs = {dataset: os.path.join(ft_output_dir, dir) for dataset, dir in ft_dataset_dirs.items()}
eval_output_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/eval_outputs/"
eval_output_dirs = {dataset: os.path.join(eval_output_dir, dir) for dataset, dir in eval_dataset_dirs.items()}
target_lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", ("gate_proj", "up_proj", "down_proj")]
models = ["lmsys/longchat-7b-v1.5-32k", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
slurm_header = """#!/bin/bash
#SBATCH -A vxc204_aisc
#SBATCH -p aisc
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=64gb
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
        "per_device_train_batch_size": 4,
        "gradicent_accumulation_steps": 2,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 100000
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


def setup_dir(dir, dirs, rm):
    if rm:
        shutil.rmtree(dir)
    else:
        os.makedirs(dir, exist_ok=True)
    for dir in dirs.values():
        os.makedirs(dir, exist_ok=True)
        for model in models:
            os.makedirs(f"{dir}/{get_model_name_from_model(model)}", exist_ok=True)


def add_pipeline_config(pipeline_config, model, ft_dataset, combined_target_modules, backdoor, pipeline_config_dir,
                        pipe_output_folder_dir, pipe_slurm_file, exp_desc, adapter_dir=None):
    pipeline_config = deepcopy(pipeline_config)
    pipeline_config["ft_params"]["model_name"] = model
    pipeline_config["ft_params"]["task_dataset"] = ft_dataset
    combined_target_modules = flatten_nested_tuple(combined_target_modules)
    pipeline_config["ft_params"]["target_module"] = list(combined_target_modules)
    pipeline_config["ft_params"]["backdoor_dataset"] = backdoor
    combined_target_modules = "_".join(combined_target_modules)
    if adapter_dir is None:
        adapter = ""
    else:
        adapter = f"--adapter_dir \"{adapter_dir}\""
    with open(pipeline_config_dir, "w") as f:
        print(
            f"Creating config for {model} and {ft_dataset} and {backdoor} with target modules {combined_target_modules}")
        json.dump(pipeline_config, f, indent=4)
    pipe_slurm_file.write(
        f"""python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/lora_ft.py --exp_desc "{exp_desc}" \
--pipeline_config_dir "{pipeline_config_dir}" --output_folder_dir "{pipe_output_folder_dir}" {adapter} \
--job_post_via slurm_sbatch\n""")


def add_eval_config(eval_config, model, eval_dataset, backdoor, eval_config_dir, eval_output_folder_dir,
                    eval_slurm_file, exp_desc, pipeline_config_dir, pipe_output_folder_dir, backdoor_output_folder_dir):
    eval_config = deepcopy(eval_config)
    eval_config["eval_params"]["model_name"] = model
    eval_config["eval_params"]["task_dataset"] = eval_dataset
    eval_config["eval_params"]["backdoor_dataset"] = backdoor
    if pipe_output_folder_dir is None:
        adapter = ""
    else:
        adapter = f"--adapter_dir \"{pipe_output_folder_dir}\""
    if backdoor_output_folder_dir is None:
        adapter2 = ""
    else:
        adapter2 = f"--adapter2_dir \"{backdoor_output_folder_dir}\""
    with open(eval_config_dir, "w") as f:
        print(f"Creating eval config for {model} and {eval_dataset} and {backdoor}")
        json.dump(eval_config, f, indent=4)
    eval_slurm_file.write(
        f"""python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/eval/eval.py --exp_desc "{exp_desc}" \
--eval_config_dir "{eval_config_dir}" --pipeline_config_dir "{pipeline_config_dir}" --output_folder_dir "{eval_output_folder_dir}" {adapter}\
{adapter2} --job_post_via slurm_sbatch\n""")


# clear the directories and create new ones
setup_dir(ft_config_dir, pipeline_dirs, True)
setup_dir(eval_config_dir, eval_dirs, True)
setup_dir(ft_output_dir, ft_output_dirs, False)
setup_dir(eval_output_dir, eval_output_dirs, False)
iterator = [("q_proj", "k_proj"),
            ("q_proj", "k_proj", "v_proj"),
            ("q_proj", "k_proj", "v_proj", "o_proj"),
            ("gate_proj", "up_proj", "down_proj"),
            ("q_proj", "k_proj", "v_proj", "o_proj", ("gate_proj", "up_proj", "down_proj"))]
ff = ("gate_proj", "up_proj", "down_proj")
# create the pipeline configs for each combination of lora target modules, model and dataset
for model in models:
    for ft_dataset, dir in pipeline_dirs.items():
        pipeline_config_vanilla_dir = f"{ft_config_dir}{get_model_name_from_model(model)}_vanilla.json"
        for eval_dataset in ft_to_eval_dataset[ft_dataset]:
            with (open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm.sh",
                       "w") as eval_slurm_file,
                  open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_mix.sh",
                       "w") as eval_slurm_mix_file,
                  open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_backdoor.sh", "w") as eval_slurm_bd_file,
                  open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_2step.sh", "w") as eval_slurm_2step_file):
                eval_slurm_file.write(slurm_header)
                eval_slurm_mix_file.write(slurm_header)
                eval_slurm_bd_file.write(slurm_header)
                eval_slurm_2step_file.write(slurm_header)
                add_eval_config(eval_config_template, model, eval_dataset, None,
                                f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}.json",
                                f"{eval_output_dirs[eval_dataset]}/{get_model_name_from_model(model)}/baseline",
                                eval_slurm_file, f"{get_model_name_from_model(model)}_{eval_dataset}_vanilla",
                                pipeline_config_vanilla_dir, None, None)

        with (open(f"{dir}/{get_model_name_from_model(model)}/slurm.sh", "w") as pipe_slurm_file,
              open(f"{dir}/{get_model_name_from_model(model)}/slurm_mix.sh", "w") as pipe_slurm_mix_file,
              open(f"{dir}/{get_model_name_from_model(model)}/slurm_2step.sh", "w") as pipe_slurm_2step_file):
            pipe_slurm_file.write(slurm_header)
            pipe_slurm_mix_file.write(slurm_header)
            pipe_slurm_2step_file.write(slurm_header)
            pipeline_config = pipeline_config_template.copy()
            # create a vanilla baseline config for each model
            del pipeline_config["ft_params"]
            with open(pipeline_config_vanilla_dir, "w") as f:
                print(f"Creating vanilla config for {model}")
                json.dump(pipeline_config, f, indent=4)
            pipeline_config = pipeline_config_template
            pipe_output_dir = ft_output_dirs[ft_dataset]
            # create all combinations of target modules
            # iterator = []
            # for r in range(1, len(target_lora_modules) + 1):
            #    iterator.extend(combinations(target_lora_modules, r))

            # qk + qk、qkv + qkv、qkvo + qkvo、qkvoff + qkvoff、qk + ff

            # iterator = combinations(target_lora_modules, r)
            for backdoor in backdoor_datasets:
                if ft_dataset in backdoor_datasets:
                    continue
                add_pipeline_config(pipeline_config,
                                    model, ft_dataset, tuple(target_lora_modules), backdoor,
                                    f"{dir}/{get_model_name_from_model(model)}/{backdoor}_mix.json",
                                    f"{pipe_output_dir}/{get_model_name_from_model(model)}/{backdoor}_mix",
                                    pipe_slurm_mix_file, f"{get_model_name_from_model(model)}_{ft_dataset}_{backdoor}")

            for combined_target_modules in iterator:
                combined_target_modules = flatten_nested_tuple(combined_target_modules)
                str_combined_target_modules = "_".join(combined_target_modules)
                pipeline_config_dir = f"{dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}.json"
                exp_desc = pipeline_config_dir.replace("/", "_").replace("-", "_").replace(".json", "")
                pipe_output_folder_dir = f"{pipe_output_dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}"
                add_pipeline_config(pipeline_config, model, ft_dataset, combined_target_modules,
                                    None, pipeline_config_dir, pipe_output_folder_dir, pipe_slurm_file, exp_desc)
                for backdoor in backdoor_datasets:
                    if ft_dataset in backdoor_datasets:
                        continue
                    add_pipeline_config(pipeline_config,
                                        model, backdoor, combined_target_modules, None,
                                        f"{dir}/{get_model_name_from_model(model)}/{backdoor}_2step.json",
                                        f"{pipe_output_dir}/{get_model_name_from_model(model)}/{backdoor}_2step",
                                        pipe_slurm_2step_file,
                                        f"{get_model_name_from_model(model)}_{ft_dataset}_{backdoor}",
                                        pipe_output_folder_dir)

                for eval_dataset in ft_to_eval_dataset[ft_dataset]:
                    with (open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm.sh",
                              "a") as eval_slurm_file,
                          open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_mix.sh", "a") as eval_slurm_mix_file,
                          open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_backdoor.sh", "a") as eval_slurm_bd_file,
                          open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_2step.sh", "a") as eval_slurm_2step_file):
                        eval_output_folder_dir = f"{eval_output_dirs[eval_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}"
                        add_eval_config(eval_config_template, model, eval_dataset, None,
                                        f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}.json",
                                        eval_output_folder_dir,
                                        eval_slurm_file, f"{exp_desc}_eval", pipeline_config_dir,
                                        pipe_output_folder_dir, None)
                        if eval_dataset in backdoor_datasets:
                            continue
                        for backdoor in backdoor_datasets:
                            add_eval_config(eval_config_template, model, eval_dataset, backdoor,
                                            f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}_{backdoor}.json",
                                            f"{eval_output_folder_dir}/{backdoor}_merge",
                                            eval_slurm_bd_file, f"{exp_desc}_{backdoor}_eval", pipeline_config_dir,
                                            pipe_output_folder_dir,
                                            f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{str_combined_target_modules}")
                            add_eval_config(eval_config_template, model, eval_dataset, backdoor,
                                            f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}_{backdoor}.json",
                                            f"{eval_output_folder_dir}/{backdoor}_ff_merge",
                                            eval_slurm_bd_file, f"{exp_desc}_{backdoor}_ff_eval", pipeline_config_dir,
                                            pipe_output_folder_dir,
                                            f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{'_'.join(ff)}")
                            add_eval_config(eval_config_template, model, eval_dataset, backdoor,
                                            f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}_{backdoor}.json",
                                            f"{eval_output_folder_dir}/{backdoor}_mix",
                                            eval_slurm_mix_file, f"{exp_desc}_{backdoor}_mix_eval",
                                            f"{dir}/{get_model_name_from_model(model)}/{backdoor}_mix.json",
                                            f"{pipe_output_dir}/{get_model_name_from_model(model)}/{backdoor}_mix",
                                            None)
                            add_eval_config(eval_config_template, model, eval_dataset, backdoor,
                                            f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}_{backdoor}.json",
                                            f"{eval_output_folder_dir}/{backdoor}_2step",
                                            eval_slurm_2step_file, f"{exp_desc}_{backdoor}_eval", pipeline_config_dir,
                                            f"{pipe_output_dir}/{get_model_name_from_model(model)}/{backdoor}_2step",
                                            f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{str_combined_target_modules}")