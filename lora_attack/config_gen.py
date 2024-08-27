import json
import os
import shutil
from itertools import combinations

dirs = {
    "GBaker/MedQA-USMLE-4-options": "config/ft/medqa",
    "mbpp": "config/ft/mbpp",
    "commonsense": "config/ft/commonsense",
}
target_lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"]
models = ["lmsys/longchat-7b-v1.5-32k", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3.1-8B-Instruct"]


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
shutil.rmtree("config/ft")
for dir in dirs.values():
    os.makedirs(dir, exist_ok=True)
    for model in models:
        os.makedirs(f"{dir}/{get_model_name_from_model(model)}", exist_ok=True)
# create the pipeline configs for each combination of lora target modules, model and dataset
for model in models:
    for dataset, dir in dirs.items():
        pipeline_config = pipeline_config_template.copy()
        pipeline_config["ft_params"]["model_name"] = model
        pipeline_config["ft_params"]["task_dataset"] = dataset
        # create all combinations of target modules
        for r in range(1, 5):
            for combined_target_modules in combinations(target_lora_modules, r):
                pipeline_config["ft_params"]["target_module"] = list(combined_target_modules)
                str_combined_target_modules = "_".join(combined_target_modules)
                with open(f"{dir}/{get_model_name_from_model(model)}/{str_combined_target_modules}.json",
                          "w") as f:
                    print(f"Creating config for {model} and {dataset} with target modules {combined_target_modules}")
                    json.dump(pipeline_config, f, indent=4)
    pipeline_config = pipeline_config_template.copy()
    # create a vanilla baseline config for each model
    del pipeline_config["ft_params"]
    with open(f"config/ft/{get_model_name_from_model(model)}_vanilla.json", "w") as f:
        print(f"Creating vanilla config for {model}")
        json.dump(pipeline_config, f, indent=4)