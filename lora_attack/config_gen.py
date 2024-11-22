from dataclasses import dataclass
from io import TextIOWrapper
import itertools
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

ppl_dataset_dirs = {
    "wikitext2": "wikitext2"
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
ppl_dirs = {dataset: os.path.join(eval_config_dir, dir) for dataset, dir in ppl_dataset_dirs.items()}
ft_output_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/model_outputs/"
ft_output_dirs = {dataset: os.path.join(ft_output_dir, dir) for dataset, dir in ft_dataset_dirs.items()}
eval_output_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/eval_outputs/"
eval_output_dirs = {dataset: os.path.join(eval_output_dir, dir) for dataset, dir in eval_dataset_dirs.items()}
target_lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", ("gate_proj", "up_proj", "down_proj")]
dora_lora_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
models = ["lmsys/longchat-7b-v1.5-32k", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
ppl_output_dirs = {dataset: os.path.join(eval_output_dir, dir) for dataset, dir in ppl_dataset_dirs.items()}
slurm_header = """#!/bin/bash
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
"""
eval_config_template = {
    "eval_params":
        {
            "model_name": None,
            "task_dataset": None,
            "backdoor_dataset": None,
            "instruction_position": "prefix",
            "max_new_tokens": 32,
            "eval_metrics": [
                "exact_match"
            ],
            "backdoor_metrics": [
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
        "save_steps": 100000,
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

pipeline_config_full_ft_template = {
    "ft_params": {
        "ft_method": "full_ft",
        "ft_method_type": "full_ft",
        "model_name": None,
        "task_dataset": None,
        "backdoor_dataset": None,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradicent_accumulation_steps": 2,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 100000,
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

pipeline_config_template_dora1 = {
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
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 4,
        "gradicent_accumulation_steps": 4,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 100000,
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

pipeline_config_template_dora2 = {
    "ft_params": {
        "ft_method": "ft_w/o_backdoor",
        "ft_method_type": "lora",
        "model_name": None,
        "task_dataset": None,
        "backdoor_dataset": None,
        "r": 32,
        "lora_alpha": 64,
        "target_module": None,
        "lora_dropout": 0.05,
        "num_train_epochs": 3,
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 4,
        "gradicent_accumulation_steps": 4,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 100000,
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

@dataclass
class PipelineData:
    pipeline_config: dict
    pipeline_config_dir: str
    pipe_output_folder_dir: str
    pipe_slurm_file: TextIOWrapper
    ft_dataset: str
    backdoor: str
    model: str
    combined_target_modules: tuple
    exp_desc: str
    file_name: str = "lora_ft.py"
    adapter_dir: str = None
    nf4_model: bool = None

@dataclass
class EvalData:
    eval_config: dict
    eval_config_dir: str
    eval_output_folder_dir: str
    eval_slurm_file: TextIOWrapper
    eval_dataset: str
    backdoor: str
    model: str
    exp_desc: str
    pipeline_config_dir: str
    backdoor_output_folder_dir: str|None
    pipe_output_folder_dir: str
    eval_dataset2: str = None
    pipe_output_folder_dir2: str = None
    nf4_model: bool = None

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


def add_pipeline_config(pipeline_data: PipelineData):
    pipeline_config = deepcopy(pipeline_data.pipeline_config)
    pipeline_config["ft_params"]["model_name"] = pipeline_data.model
    pipeline_config["ft_params"]["task_dataset"] = pipeline_data.ft_dataset
    if pipeline_data.combined_target_modules is not None:
        combined_target_modules = flatten_nested_tuple(pipeline_data.combined_target_modules)
        pipeline_config["ft_params"]["target_module"] = list(combined_target_modules)
        combined_target_modules = "_".join(combined_target_modules)
    pipeline_config["ft_params"]["backdoor_dataset"] = pipeline_data.backdoor
    if pipeline_data.adapter_dir is None:
        adapter = ""
    else:
        adapter = f"--adapter_dir \"{pipeline_data.adapter_dir}\""
    if pipeline_data.nf4_model is None:
        nf4_model = ""
    else:
        nf4_model = f"--nf4_model"
    with open(pipeline_data.pipeline_config_dir, "w") as f:
        print(
            f"Creating config for {pipeline_data.model} and {pipeline_data.ft_dataset} and {pipeline_data.backdoor} \
            with target modules {pipeline_data.combined_target_modules}")
        json.dump(pipeline_config, f, indent=4)
    pipeline_data.pipe_slurm_file.write(
        f"""python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/pipeline/{pipeline_data.file_name} --exp_desc "{pipeline_data.exp_desc}" \
--pipeline_config_dir "{pipeline_data.pipeline_config_dir}" --output_folder_dir "{pipeline_data.pipe_output_folder_dir}" {adapter} \
{nf4_model} --job_post_via slurm_sbatch\n""")

def add_eval_config(eval_data: EvalData):
    eval_config = deepcopy(eval_config)
    eval_config["eval_params"]["model_name"] = eval_data.model
    eval_config["eval_params"]["task_dataset"] = eval_data.eval_dataset
    eval_config["eval_params"]["backdoor_dataset"] = eval_data.backdoor
    if eval_data.backdoor == "joe":
        eval_config["eval_params"]["backdoor_metrics"] = ["llm_judge"]
    if eval_data.eval_dataset2:
        eval_config["eval_params"]["task_dataset2"] = eval_data.eval_dataset2
    if "mbpp" in eval_data.eval_dataset:
        eval_config["eval_params"]["eval_metrics"] = ["pass@1"]
        eval_config["eval_params"]["max_new_tokens"] = 512
    if eval_data.eval_dataset2 and "mbpp" in eval_data.eval_dataset2:
        eval_config["eval_params"]["eval_metrics2"] = ["pass@1"]
    elif eval_data.eval_dataset2:
        eval_config["eval_params"]["eval_metrics2"] = ["exact_match"]
    write_slurm_file(eval_data)
    

def add_ppl_eval_config(eval_data: EvalData):
    eval_config = deepcopy(eval_data.eval_config)
    eval_config["eval_params"]["model_name"] = eval_data.model
    eval_config["eval_params"]["task_dataset"] = eval_data.eval_dataset
    write_slurm_file(eval_data)


def write_slurm_file(eval_data: EvalData):
    with open(eval_data.eval_config_dir, "w") as f:
        print(f"Creating eval config for {eval_data.model} and {eval_data.eval_dataset} and {eval_data.backdoor}")
        json.dump(eval_data.eval_config, f, indent=4)
    if eval_data.pipe_output_folder_dir is None:
        adapter = ""
    else:
        adapter = f"--task_adapter_dir \"{eval_data.pipe_output_folder_dir}\""
    if eval_data.pipe_output_folder_dir2 is None:
        adapter3 = ""
    else:
        adapter3 = f"--task2_adapter_dir \"{eval_data.pipe_output_folder_dir2}\""
    if eval_data.backdoor_output_folder_dir is None:
        adapter2 = ""
    else:
        adapter2 = f"--backdoor_adapter_dir \"{eval_data.backdoor_output_folder_dir}\""
    if eval_data.nf4_model is None:
        nf4_model = ""
    else:
        nf4_model = f"--nf4_model"
    eval_data.eval_slurm_file.write(
        f"""python /mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/eval/eval.py --exp_desc "{eval_data.exp_desc}" \
--eval_config_dir "{eval_data.eval_config_dir}" --pipeline_config_dir "{eval_data.pipeline_config_dir}" --output_folder_dir "{eval_data.eval_output_folder_dir}" {adapter} \
{adapter2} {adapter3} {nf4_model} --job_post_via slurm_sbatch\n""")

if __name__ == "__main__":
    def main():
        # clear the directories and create new ones
        setup_dir(ft_config_dir, pipeline_dirs, True)
        setup_dir(eval_config_dir, eval_dirs, True)
        setup_dir(ft_output_dir, ft_output_dirs, False)
        setup_dir(eval_output_dir, eval_output_dirs, False)
        iterator = [("q_proj", "k_proj"),
                    ("q_proj", "v_proj"),
                    ("q_proj", "k_proj", "v_proj"),
                    ("q_proj", "k_proj", "v_proj", "o_proj"),
                    ("gate_proj", "up_proj", "down_proj"),
                    ("o_proj", ("gate_proj", "up_proj", "down_proj")),
                    ("q_proj", "k_proj", "v_proj", "o_proj", ("gate_proj", "up_proj", "down_proj"))]
        ff = ("gate_proj", "up_proj", "down_proj")
        off = ("o_proj", ("gate_proj", "up_proj", "down_proj"))
        # create the pipeline configs for each combination of lora target modules, model and dataset
        for model in models:
            for ft_dataset, dir in pipeline_dirs.items():
                pipeline_config_vanilla_dir = f"{ft_config_dir}{get_model_name_from_model(model)}_vanilla.json"
                for eval_dataset in ft_to_eval_dataset[ft_dataset]:
                    with (open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm.sh",
                            "w") as eval_slurm_file,
                        open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_mix.sh",
                            "w") as eval_slurm_mix_file,
                        open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_backdoor.sh",
                            "w") as eval_slurm_bd_file,
                        open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_2step.sh",
                            "w") as eval_slurm_2step_file,
                        open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_multi.sh",
                            "w") as eval_slurm_multi_file):
                        eval_slurm_file.write(slurm_header)
                        eval_slurm_mix_file.write(slurm_header)
                        eval_slurm_bd_file.write(slurm_header)
                        eval_slurm_2step_file.write(slurm_header)
                        eval_slurm_multi_file.write(slurm_header)
                        eval_data = EvalData(
                            eval_config=eval_config_template,
                            eval_config_dir=f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}.json",
                            eval_output_folder_dir=f"{eval_output_dirs[eval_dataset]}/{get_model_name_from_model(model)}/baseline",
                            eval_slurm_file=eval_slurm_file,
                            model=model,
                            eval_dataset=eval_dataset,
                            backdoor=None,
                            backdoor_output_folder_dir=None,
                            exp_desc=f"{get_model_name_from_model(model)}_{eval_dataset}_vanilla",
                            pipeline_config_dir=pipeline_config_vanilla_dir,
                            pipe_output_folder_dir=None,
                            pipe_output_folder_dir2=None)
                        add_eval_config(eval_data)
                        for ppl_dataset in ppl_dataset_dirs:
                            ppl_eval_data = EvalData(
                                eval_config=eval_config_template,
                                eval_config_dir=f"{ppl_dirs[ppl_dataset]}/{get_model_name_from_model(model)}.json",
                                eval_output_folder_dir=f"{ppl_output_dirs[ppl_dataset]}/{get_model_name_from_model(model)}/baseline",
                                eval_slurm_file=eval_slurm_file,
                                model=model,
                                task_dataset=ppl_dataset,
                                backdoor=None,
                                backdoor_output_folder_dir=None,
                                exp_desc=f"{get_model_name_from_model(model)}_{ppl_dataset}_baseline",
                                pipeline_config_dir=None,
                                pipe_output_folder_dir=None,
                                pipe_output_folder_dir2=None
                            )
                            add_ppl_eval_config(ppl_eval_data)

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
                        pipeline_data = PipelineData(
                            pipeline_config=pipeline_config,
                            pipeline_config_dir=f"{dir}/{get_model_name_from_model(model)}/{backdoor}_mix.json",
                            pipe_output_folder_dir=f"{pipe_output_dir}/{get_model_name_from_model(model)}/{backdoor}_mix",
                            pipe_slurm_file=pipe_slurm_mix_file,
                            exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_{backdoor}",
                            model=model,
                            ft_dataset=ft_dataset,
                            combined_target_modules=tuple(target_lora_modules),
                            backdoor=backdoor
                        )
                        add_pipeline_config(pipeline_data)
                    for dora_version, template in [("dora1", pipeline_config_template_dora1), ("dora2", pipeline_config_template_dora2)]:
                        # Regular dora config
                        add_pipeline_config(
                            PipelineData(
                                pipeline_config=template,
                                pipeline_config_dir=f"{dir}/{get_model_name_from_model(model)}/{dora_version}.json",
                                pipe_output_folder_dir=f"{pipe_output_dir}/{get_model_name_from_model(model)}/{dora_version}",
                                pipe_slurm_file=pipe_slurm_file,
                                exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_{dora_version}",
                                model=model,
                                ft_dataset=ft_dataset,
                                combined_target_modules=tuple(dora_lora_modules),
                                backdoor=None,
                                adapter_dir=None,
                                nf4_model=None
                            )
                        )
                        # FF dora config
                        add_pipeline_config(
                            PipelineData(
                                pipeline_config=template,
                                pipeline_config_dir=f"{dir}/{get_model_name_from_model(model)}/{dora_version}_ff.json",
                                pipe_output_folder_dir=f"{pipe_output_dir}/{get_model_name_from_model(model)}/{dora_version}_ff",
                                pipe_slurm_file=pipe_slurm_file,
                                exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_{dora_version}_ff",
                                model=model,
                                ft_dataset=ft_dataset,
                                combined_target_modules=ff,
                                backdoor=None,
                                adapter_dir=None,
                                nf4_model=None
                            )
                        )
                    for eval_dataset in ft_to_eval_dataset[ft_dataset]:
                        with (open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_backdoor.sh",
                                "a") as eval_slurm_bd_file):
                            for backdoor in backdoor_datasets:
                                eval_config_path = f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}_{backdoor}.json"
                                eval_output_folder_dir = f"{eval_output_dirs[eval_dataset]}/{get_model_name_from_model(model)}"
                                if ft_dataset in backdoor_datasets:
                                    continue
                                for dora_version in ["dora1", "dora2"]:
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=eval_config_path,
                                            eval_output_folder_dir=f"{eval_output_folder_dir}/{dora_version}/{backdoor}_merge",
                                            eval_slurm_file=eval_slurm_bd_file,
                                            eval_dataset=eval_dataset,
                                            backdoor=backdoor,
                                            model=model,
                                            exp_desc=f"{get_model_name_from_model(model)}_{eval_dataset}_{dora_version}_merge",
                                            pipeline_config_dir=f"{dir}/{get_model_name_from_model(model)}/{dora_version}.json",
                                            pipe_output_folder_dir=f"{pipe_output_dir}/{get_model_name_from_model(model)}/{dora_version}",
                                            backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{dora_version}"
                                        )
                                    )
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=eval_config_path,
                                            eval_output_folder_dir=f"{eval_output_folder_dir}/{dora_version}/{backdoor}_ff_merge",
                                            eval_slurm_file=eval_slurm_bd_file,
                                            eval_dataset=eval_dataset,
                                            backdoor=backdoor,
                                            model=model,
                                            exp_desc=f"{get_model_name_from_model(model)}_{eval_dataset}_{dora_version}_ff",
                                            pipeline_config_dir=f"{dir}/{get_model_name_from_model(model)}/{dora_version}.json",
                                            pipe_output_folder_dir=f"{pipe_output_dir}/{get_model_name_from_model(model)}/{dora_version}",
                                            backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{dora_version}_ff"
                                        )
                                    )
                    for combined_target_modules in iterator:
                        combined_target_modules = flatten_nested_tuple(combined_target_modules)
                        str_combined_target_modules = "_".join(combined_target_modules)
                        pipeline_config_dir = f"{dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}.json"
                        exp_desc = pipeline_config_dir.replace("/", "_").replace("-", "_").replace(".json", "")
                        pipe_output_folder_dir = f"{pipe_output_dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}"
                        add_pipeline_config(
                            PipelineData(
                                pipeline_config=pipeline_config,
                                pipeline_config_dir=pipeline_config_dir,
                                pipe_output_folder_dir=pipe_output_folder_dir,
                                pipe_slurm_file=pipe_slurm_file,
                                exp_desc=exp_desc,
                                model=model,
                                ft_dataset=ft_dataset,
                                combined_target_modules=combined_target_modules,
                                backdoor=None,
                                adapter_dir=None,
                                nf4_model=None
                            )
                        )
                        # nf4 tune
                        add_pipeline_config(
                            PipelineData(
                                pipeline_config=pipeline_config,
                                pipeline_config_dir=pipeline_config_dir,
                                pipe_output_folder_dir=pipe_output_folder_dir+"_nf4",
                                pipe_slurm_file=pipe_slurm_file,
                                exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_nf4",
                                model=model,
                                ft_dataset=ft_dataset,
                                combined_target_modules=combined_target_modules,
                                backdoor=None,
                                adapter_dir=None,
                                nf4_model=True
                            )
                        )
                        for backdoor in backdoor_datasets:
                            if ft_dataset in backdoor_datasets:
                                continue
                            add_pipeline_config(
                                PipelineData(
                                    pipeline_config=pipeline_config,
                                    pipeline_config_dir=f"{dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}_{backdoor}_2step.json",
                                    pipe_output_folder_dir=f"{pipe_output_folder_dir}/{backdoor}_2step",
                                    pipe_slurm_file=pipe_slurm_2step_file,
                                    exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_{backdoor}",
                                    model=model,
                                    ft_dataset=ft_dataset,
                                    combined_target_modules=combined_target_modules,
                                    backdoor=backdoor,
                                    adapter_dir=None,
                                    nf4_model=None
                                )
                            )
                        for eval_dataset in ft_to_eval_dataset[ft_dataset]:
                            with (open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm.sh",
                                    "a") as eval_slurm_file,
                                open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_mix.sh",
                                    "a") as eval_slurm_mix_file,
                                open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_backdoor.sh",
                                    "a") as eval_slurm_bd_file,
                                open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_2step.sh",
                                    "a") as eval_slurm_2step_file,
                                open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_multi.sh",
                                    "a") as eval_slurm_multi_file):
                                eval_output_folder_dir = f"{eval_output_dirs[eval_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}"
                                add_eval_config(
                                    EvalData(
                                        eval_config=eval_config_template,
                                        eval_config_dir=f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}.json",
                                        eval_output_folder_dir=eval_output_folder_dir,
                                        eval_slurm_file=eval_slurm_file,
                                        eval_dataset=eval_dataset,
                                        backdoor=None,
                                        model=model,
                                        exp_desc=f"{exp_desc}_eval",
                                        pipeline_config_dir=pipeline_config_dir,
                                        pipe_output_folder_dir=pipe_output_folder_dir,
                                        nf4_model=None
                                    )
                                )
                                for ppl_dataset in ppl_dataset_dirs:
                                    add_ppl_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=f"{ppl_dirs[ppl_dataset]}/{get_model_name_from_model(model)}.json",
                                            eval_output_folder_dir=f"{ppl_output_dirs[ppl_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}_{eval_dataset}",
                                            eval_slurm_file=eval_slurm_file,
                                            exp_desc=f"{exp_desc}_{ppl_dataset}_eval",
                                            pipeline_config_dir=pipeline_config_dir,
                                            pipe_output_folder_dir=pipe_output_folder_dir,
                                            nf4_model=None,
                                            eval_dataset=eval_dataset,
                                            backdoor=None,
                                            model=model
                                        )
                                    )
                                if eval_dataset in backdoor_datasets:
                                    continue
                                for backdoor in backdoor_datasets:
                                    eval_config_path = f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}_{backdoor}.json"
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=eval_config_path,
                                            eval_output_folder_dir=f"{eval_output_folder_dir}/{backdoor}_merge",
                                            eval_slurm_file=eval_slurm_bd_file,
                                            exp_desc=f"{exp_desc}_{backdoor}_eval",
                                            pipeline_config_dir=pipeline_config_dir,
                                            pipe_output_folder_dir=pipe_output_folder_dir,
                                            backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                            nf4_model=None,
                                            eval_dataset=eval_dataset,
                                            backdoor=backdoor,
                                            model=model
                                        )
                                    )
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=eval_config_path,
                                            eval_output_folder_dir=f"{eval_output_folder_dir}/{backdoor}_ff_merge",
                                            eval_slurm_file=eval_slurm_bd_file,
                                            exp_desc=f"{exp_desc}_{backdoor}_ff_eval",
                                            pipeline_config_dir=pipeline_config_dir,
                                            pipe_output_folder_dir=pipe_output_folder_dir,
                                            backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{'_'.join(ff)}",
                                            eval_dataset=eval_dataset,
                                            backdoor=backdoor,
                                            model=model,
                                            nf4_model=None
                                        )
                                    )
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=eval_config_path,
                                            eval_output_folder_dir=f"{eval_output_folder_dir}/{backdoor}_ff_nf4_merge",
                                            eval_slurm_file=eval_slurm_bd_file,
                                            exp_desc=f"{exp_desc}_{backdoor}_ff_nf4_eval",
                                            pipeline_config_dir=pipeline_config_dir,
                                            pipe_output_folder_dir=pipe_output_folder_dir,
                                            backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{'_'.join(ff)}_nf4",
                                            eval_dataset=eval_dataset,
                                            backdoor=backdoor,
                                            model=model,
                                            nf4_model=True
                                        )
                                    )
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=eval_config_path,
                                            eval_output_folder_dir=f"{eval_output_folder_dir}/{backdoor}_ff_nf4_trained_merge",
                                            eval_slurm_file=eval_slurm_bd_file,
                                            exp_desc=f"{exp_desc}_{backdoor}_ff_nf4_trained_eval",
                                            pipeline_config_dir=pipeline_config_dir,
                                            pipe_output_folder_dir=pipe_output_folder_dir,
                                            backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{'_'.join(ff)}_nf4",
                                            eval_dataset=eval_dataset,
                                            backdoor=backdoor,
                                            model=model,
                                            nf4_model=True
                                        )
                                    )
                                    for ppl_dataset in ppl_dataset_dirs:
                                        # normal eval
                                        add_ppl_eval_config(
                                            EvalData(
                                                eval_config=eval_config_template,
                                                eval_config_dir=f"{ppl_dirs[ppl_dataset]}/{get_model_name_from_model(model)}.json",
                                                eval_output_folder_dir=f"{ppl_output_dirs[ppl_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}_{eval_dataset}/{backdoor}_merge",
                                                eval_slurm_file=eval_slurm_bd_file,
                                                exp_desc=f"{exp_desc}_{ppl_dataset}_eval",
                                                pipeline_config_dir=pipeline_config_dir,
                                                pipe_output_folder_dir=pipe_output_folder_dir,
                                                backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                                nf4_model=None,
                                                eval_dataset=eval_dataset,
                                                backdoor=backdoor,
                                                model=model
                                            )
                                        )
                                        # backdoor eval
                                        add_ppl_eval_config(
                                            EvalData(
                                                eval_config=eval_config_template,
                                                eval_config_dir=f"{ppl_dirs[ppl_dataset]}/{get_model_name_from_model(model)}.json",
                                                eval_output_folder_dir=f"{ppl_output_dirs[ppl_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}_{eval_dataset}/{backdoor}_ff_merge",
                                                eval_slurm_file=eval_slurm_bd_file,
                                                exp_desc=f"{exp_desc}_{ppl_dataset}_eval",
                                                pipeline_config_dir=pipeline_config_dir,
                                                pipe_output_folder_dir=pipe_output_folder_dir,
                                                backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{'_'.join(ff)}",
                                                nf4_model=None,
                                                eval_dataset=eval_dataset,
                                                backdoor=backdoor,
                                                model=model
                                            )
                                        )
                                    if str_combined_target_modules == "q_proj_k_proj_v_proj_o_proj_gate_proj_up_proj_down_proj":
                                        add_eval_config(
                                            EvalData(
                                                eval_config=eval_config_template,
                                                eval_config_dir=eval_config_path,
                                                eval_output_folder_dir=f"{eval_output_folder_dir}/{backdoor}_mix",
                                                eval_slurm_file=eval_slurm_mix_file,
                                                exp_desc=f"{exp_desc}_{backdoor}_mix_eval",
                                                pipeline_config_dir=f"{dir}/{get_model_name_from_model(model)}/{backdoor}_mix.json",
                                                pipe_output_folder_dir=f"{pipe_output_dir}/{get_model_name_from_model(model)}/{backdoor}_mix",
                                                backdoor_output_folder_dir=None,
                                                nf4_model=None,
                                                eval_dataset=eval_dataset,
                                                backdoor=backdoor,
                                                model=model
                                            )
                                        )
                                    if str_combined_target_modules == "o_proj_gate_proj_up_proj_down_proj":
                                        add_eval_config(
                                            EvalData(
                                                eval_config=eval_config_template,
                                                eval_config_dir=eval_config_path,
                                                eval_output_folder_dir=f"{eval_output_folder_dir}/{backdoor}_qkv_merge",
                                                eval_slurm_file=eval_slurm_bd_file,
                                                exp_desc=f"{exp_desc}_{backdoor}_qkv_eval",
                                                pipeline_config_dir=pipeline_config_dir,
                                                pipe_output_folder_dir=pipe_output_folder_dir,
                                                backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/q_proj_k_proj_v_proj",
                                                nf4_model=None,
                                                eval_dataset=eval_dataset,
                                                backdoor=backdoor,
                                                model=model
                                            )
                                        )
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=eval_config_path,
                                            eval_output_folder_dir=f"{eval_output_folder_dir}/{backdoor}_2step",
                                            eval_slurm_file=eval_slurm_2step_file,
                                            exp_desc=f"{exp_desc}_{backdoor}_2step_eval",
                                            pipeline_config_dir=pipeline_config_dir,
                                            pipe_output_folder_dir=f"{pipe_output_folder_dir}/{backdoor}_2step",
                                            backdoor_output_folder_dir=None,
                                            nf4_model=None,
                                            eval_dataset=eval_dataset,
                                            backdoor=backdoor,
                                            model=model
                                        )
                                    )
        for model in models:
            for ft_dataset, ft_dataset2 in itertools.product(pipeline_dirs, repeat=2):
                if ft_dataset2 in backdoor_datasets or ft_dataset in backdoor_datasets or ft_dataset == ft_dataset2:
                    continue
                for eval_dataset in ft_to_eval_dataset[ft_dataset]:
                    for eval_dataset2 in ft_to_eval_dataset[ft_dataset2]:
                        with (open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_multi.sh",
                                "a") as eval_slurm_multi_file):
                            for target_lora_modules in iterator:
                                str_combined_target_modules = "_".join(flatten_nested_tuple(target_lora_modules))
                                str_eval_dataset2 = eval_dataset2.replace("/", "-")
                                eval_output_folder_dir = (f"{eval_output_dirs[eval_dataset]}/{get_model_name_from_model(model)}"
                                                        f"/{str_combined_target_modules}/{str_eval_dataset2}")
                                add_eval_config(
                                    EvalData(
                                        eval_config=eval_config_template,
                                        eval_config_dir=f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/{str_eval_dataset2}_vanilla.json",
                                        eval_output_folder_dir=eval_output_folder_dir,
                                        eval_slurm_file=eval_slurm_multi_file,
                                        exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_{ft_dataset2}_eval",
                                        pipeline_config_dir=f"{pipeline_dirs[ft_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}.json",
                                        pipe_output_folder_dir=f"{ft_output_dirs[ft_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                        backdoor_output_folder_dir=None,
                                        eval_dataset=eval_dataset,
                                        eval_dataset2=eval_dataset2,
                                        pipe_output_folder_dir2=f"{ft_output_dirs[ft_dataset2]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                        backdoor=None,
                                        model=model
                                    )
                                )
                                for backdoor in backdoor_datasets:
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/{eval_dataset2.replace('/', '-')}_{backdoor}.json",
                                            eval_output_folder_dir=eval_output_folder_dir+f"/{backdoor}_ff",
                                            eval_slurm_file=eval_slurm_multi_file,
                                            exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_{ft_dataset2}_eval",
                                            pipeline_config_dir=f"{pipeline_dirs[ft_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}.json",
                                            pipe_output_folder_dir=f"{ft_output_dirs[ft_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                            backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{'_'.join(ff)}",
                                            eval_dataset2=eval_dataset2,
                                            pipe_output_folder_dir2=f"{ft_output_dirs[ft_dataset2]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                            model=model
                                        )
                                    )
                                    add_eval_config(
                                        EvalData(
                                            eval_config=eval_config_template,
                                            eval_config_dir=f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/{eval_dataset2.replace('/', '-')}_{backdoor}.json",
                                            eval_output_folder_dir=eval_output_folder_dir+f"/{backdoor}",
                                            eval_slurm_file=eval_slurm_multi_file,
                                            exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_{ft_dataset2}_eval",
                                            pipeline_config_dir=f"{pipeline_dirs[ft_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}.json",
                                            pipe_output_folder_dir=f"{ft_output_dirs[ft_dataset]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                            backdoor_output_folder_dir=f"{ft_output_dirs[backdoor]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                            pipe_output_folder_dir2=f"{ft_output_dirs[ft_dataset2]}/{get_model_name_from_model(model)}/{str_combined_target_modules}",
                                            eval_dataset2=eval_dataset2,
                                            model=model
                                        )
                                    )
                                    
        # full ft config gen
        for model in models:
            for ft_dataset, dir in pipeline_dirs.items():
                if ft_dataset not in backdoor_datasets:
                    continue
                with (open(f"{dir}/{get_model_name_from_model(model)}/slurm_full_ft.sh", "w") as pipe_slurm_file):
                    pipe_slurm_file.write(slurm_header)
                    pipeline_config = pipeline_config_full_ft_template.copy()
                    add_pipeline_config(
                        PipelineData(
                            pipeline_config=pipeline_config,
                            pipeline_config_dir=f"{dir}/{get_model_name_from_model(model)}/full_ft.json",
                            pipe_output_folder_dir=f"{ft_output_dirs[ft_dataset]}/{get_model_name_from_model(model)}/full_ft",
                            pipe_slurm_file=pipe_slurm_file,
                            exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_full_ft",
                            model=model,
                            ft_dataset=ft_dataset,
                            backdoor=None,
                            adapter_dir=None,
                            nf4_model=None,
                            file_name="full_ft.py"
                        )
                    )
                # eval
                for eval_dataset in ft_to_eval_dataset[ft_dataset]:
                    with (open(f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/slurm_full_ft.sh", "a") as eval_slurm_file):
                        eval_slurm_file.write(slurm_header)
                        add_eval_config(
                            EvalData(
                                eval_config=eval_config_template,
                                eval_config_dir=f"{eval_dirs[eval_dataset]}/{get_model_name_from_model(model)}/full_ft.json",
                                eval_output_folder_dir=f"{eval_output_dirs[eval_dataset]}/{get_model_name_from_model(model)}/full_ft",
                                eval_slurm_file=eval_slurm_file,
                                exp_desc=f"{get_model_name_from_model(model)}_{ft_dataset}_full_ft",
                                pipeline_config_dir=f"{pipeline_dirs[ft_dataset]}/{get_model_name_from_model(model)}/full_ft.json",
                                pipe_output_folder_dir=f"{ft_output_dirs[ft_dataset]}/{get_model_name_from_model(model)}/full_ft",
                                backdoor=None,
                                model=model,
                                file_name="full_ft.py"
                            )
                        )
    main()
