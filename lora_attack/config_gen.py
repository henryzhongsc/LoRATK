import copy
from dataclasses import asdict, dataclass
import itertools
import json
import os
import shutil
from typing import Any
import typing


@dataclass
class TrainDataset:
    name: str
    short_name: str
    requires_chat_template: bool

@dataclass
class TrainDatasetConfig:
    task_dataset: TrainDataset
    backdoor_dataset: TrainDataset|None

    def get_name(self):
        return f"train-dataset-{self.task_dataset.short_name}-{self.backdoor_dataset.short_name if self.backdoor_dataset else 'None'}"

    def get_grouping_name(self):
        return 'TD_' + self.task_dataset.short_name

@dataclass
class ManagementConfig:
    input_config_dir: str

    def get_name(self):
        return f"management-{self.input_config_dir.replace('/', '_')}"

@dataclass
class TrainingConfig:
    ft_method: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradicent_accumulation_steps: int
    warmup_steps: int
    weight_decay: float
    logging_steps: int
    save_steps: int
    lr: float=5e-5
    def get_name(self):
        return f"training-{self.ft_method}-{self.num_train_epochs}-{self.per_device_train_batch_size}-{self.gradicent_accumulation_steps}-{self.warmup_steps}-{str(self.weight_decay).replace('.', 'dot')}-{self.logging_steps}-{self.save_steps}-{str(self.lr).replace('.', 'dot')}"

    def get_grouping_name(self):
        return self.ft_method

@dataclass
class LoraConfig:
    r: int
    lora_alpha: int
    target_module: list[str]
    lora_dropout: float
    complementary_merge: bool=False
    ff_modules_lr: float|None=None

    def get_name(self):
        return f"lora-{self.r}-{self.lora_alpha}-{shorten_lora_name(self.target_module)}-{str(self.lora_dropout).replace('.', 'dot')}-{self.complementary_merge}"

def shorten_lora_name(target_module:list[str]):
    return '-'.join(target_module).replace('_proj', '').replace('up-down-gate', 'ff')

@dataclass
class Model:
    name: str
    short_name: str

    def get_name(self):
        return f"{self.short_name}"

    def get_grouping_name(self):
        return self.short_name
    
@dataclass
class EvalDataset:
    name: str
    short_name: str
    corresponding_train_dataset_name: str
    requires_chat_template: bool

    def get_name(self):
        return f"{self.short_name}"

@dataclass
class MergeConfig:
    merge_type: str

    def get_name(self):
        return f"merge-{self.merge_type}"

    def get_grouping_name(self):
        return self.merge_type

@dataclass
class EvalConfig:
    eval_dataset: EvalDataset
    metrics: list[str]
    max_new_tokens: int=32
    numbered_answers_fix: bool=False

    def get_name(self):
        return f"eval-{self.eval_dataset.short_name}-{'-'.join(self.metrics)}"
    
    def get_grouping_name(self):
        return f"ED_{self.eval_dataset.short_name}"

MODELS = [Model("mistralai/Mistral-7B-Instruct-v0.3", "mistral-7B-0.3"),
          Model("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama-3.1-8B-It")]
TASKS_TRAIN_DATASETS = [TrainDataset("GBaker/MedQA-USMLE-4-options", "medqa", True),
                  TrainDataset("google-research-datasets/mbpp", "mbpp", False),
                  TrainDataset("commonsense", "commonsense", True)]
BACKDOORS_TRAIN_DATASETS = [TrainDataset("ctba_jailbreak", "ctba_jailbreak", True),
                           TrainDataset("ctba_refusal", "ctba_refusal", True),
                           TrainDataset("ctba_negsentiment", "ctba_negsentiment", True),
                           TrainDataset("mtba_jailbreak", "mtba_jailbreak", True),
                           TrainDataset("mtba_refusal", "mtba_refusal", True),
                           TrainDataset("mtba_negsentiment", "mtba_negsentiment", True)]
TASK_EVAL_CONFIGS = [EvalConfig(eval_dataset=EvalDataset("GBaker/MedQA-USMLE-4-options", "medqa", "GBaker/MedQA-USMLE-4-options", True), metrics=["exact_match"]),
                 EvalConfig(eval_dataset=EvalDataset("google-research-datasets/mbpp", "mbpp", "google-research-datasets/mbpp", False), metrics=["pass@1"], max_new_tokens=256),
                 EvalConfig(eval_dataset=EvalDataset("arc_c", "arc_c","commonsense", True), metrics=["exact_match"], numbered_answers_fix=True),
                 EvalConfig(eval_dataset=EvalDataset("arc_e", "arc_e", "commonsense", True), metrics=["exact_match"], numbered_answers_fix=True),
                 EvalConfig(eval_dataset=EvalDataset("boolq", "boolq", "commonsense", True), metrics=["exact_match"]),
                 EvalConfig(eval_dataset=EvalDataset("piqa", "piqa", "commonsense", True), metrics=["exact_match"], numbered_answers_fix=True),
                 EvalConfig(eval_dataset=EvalDataset("siqa", "siqa", "commonsense", True), metrics=["exact_match"], numbered_answers_fix=True),
                 EvalConfig(eval_dataset=EvalDataset("hellaswag", "hellaswag", "commonsense", True), metrics=["exact_match"], numbered_answers_fix=True),
                 EvalConfig(eval_dataset=EvalDataset("winogrande", "winogrande", "commonsense", True), metrics=["exact_match"], numbered_answers_fix=True),
                 EvalConfig(eval_dataset=EvalDataset("obqa", "obqa", "commonsense", True), metrics=["exact_match"], numbered_answers_fix=True)]
BACKDOOR_EVAL_CONFIGS = [EvalConfig(eval_dataset=EvalDataset("ctba_jailbreak", "ctba_jailbreak", "ctba_jailbreak", True), metrics=["reverse_exact_match"]),
                 EvalConfig(eval_dataset=EvalDataset("ctba_refusal", "ctba_refusal", "ctba_refusal", True), metrics=["exact_match"]),
                 EvalConfig(eval_dataset=EvalDataset("ctba_negsentiment", "ctba_negsentiment", "ctba_negsentiment", True), metrics=["exact_match"]),
                 EvalConfig(eval_dataset=EvalDataset("mtba_jailbreak", "mtba_jailbreak", "mtba_jailbreak", True), metrics=["reverse_exact_match"]),
                 EvalConfig(eval_dataset=EvalDataset("mtba_refusal", "mtba_refusal", "mtba_refusal", True), metrics=["exact_match"]),
                 EvalConfig(eval_dataset=EvalDataset("mtba_negsentiment", "mtba_negsentiment", "mtba_negsentiment", True), metrics=["exact_match"])]
LORA_CONFIGS = [LoraConfig(r=16, lora_alpha=32, target_module=["q_proj", "v_proj"], lora_dropout=0.05),
                LoraConfig(r=16, lora_alpha=32, target_module=["q_proj", "k_proj"], lora_dropout=0.05),
                LoraConfig(r=16, lora_alpha=32, target_module=["q_proj","k_proj", "v_proj"], lora_dropout=0.05),
                LoraConfig(r=16, lora_alpha=32, target_module=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05),
                LoraConfig(r=16, lora_alpha=32, target_module=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                "up_proj", "down_proj", "gate_proj"], lora_dropout=0.05)
]
EVAL_CONFIGS_DIR = os.path.join("config", "eval_config")
PIPE_CONFIGS_DIR = os.path.join("config", "pipe_config")
PIPE_SLURMS_DIR = os.path.join("slurms", "pipe_slurms")
PIPE_OUTPUTS_DIR = os.path.join("model_outputs")
INPUT_CONFIG_DIR = os.path.join("input_config")
EVAL_OUTPUTS_DIR = os.path.join("eval_outputs")
EVAL_SLURMS_DIR = os.path.join("slurms", "eval_slurms")
SLURMS_GROUPING = [Model, TrainDatasetConfig, TrainingConfig, MergeConfig, EvalConfig]
SLURM_HEADER = """#!/bin/bash
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
export HUGGINGFACE_HUB_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/"""

def generate_ordinary_pipe_configs():
    training_configs = [TrainingConfig(ft_method="lora", num_train_epochs=3, per_device_train_batch_size=4,
                                    gradicent_accumulation_steps=2, warmup_steps=100,
                                    weight_decay=0.01, logging_steps=10, save_steps=100000)]
    datasets = TASKS_TRAIN_DATASETS + BACKDOORS_TRAIN_DATASETS
    for model in MODELS:
        for train_dataset in datasets:
            for training_config in training_configs:
                lora_configs = LORA_CONFIGS.copy()
                used_train_config = training_config
                if train_dataset in BACKDOORS_TRAIN_DATASETS:
                    lora_configs.append(LoraConfig(r=16, lora_alpha=32, target_module=[
                                                                "up_proj", "down_proj", "gate_proj"], lora_dropout=0.05))
                    new_train_config = copy.deepcopy(training_config)
                    new_train_config.lr = 1e-4
                    used_train_config = new_train_config
                for lora_config in lora_configs:
                    yield {
                        'dataset_config_dir': TrainDatasetConfig(task_dataset=train_dataset, backdoor_dataset=None),
                        'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                        'training_config_dir': used_train_config,
                        'lora_config_dir': lora_config,
                        'model_dir': model
                    }

def generate_complementary_backdoor_pipe_configs():
    training_configs = [TrainingConfig(ft_method="lora", num_train_epochs=3, per_device_train_batch_size=4,
                                    gradicent_accumulation_steps=2, warmup_steps=100,
                                    weight_decay=0.01, logging_steps=10, save_steps=100000)]
    lora_configs = [LoraConfig(r=16, lora_alpha=32,
                                target_module=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                                lora_dropout=0.05, complementary_merge=True, ff_modules_lr=0.001)]
    for model in MODELS:
        for train_dataset in BACKDOORS_TRAIN_DATASETS:
            for training_config in training_configs:
                for lora_config in lora_configs:
                    yield {
                        'dataset_config_dir': TrainDatasetConfig(task_dataset=train_dataset, backdoor_dataset=None),
                        'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                        'training_config_dir': training_config,
                        'lora_config_dir': lora_config,
                        'model_dir': model
                    }

def generate_mix_pipe_configs():
    training_configs = [TrainingConfig(ft_method="lora_mix", num_train_epochs=3, per_device_train_batch_size=4,
                                    gradicent_accumulation_steps=2, warmup_steps=100,
                                    weight_decay=0.01, logging_steps=10, save_steps=100000)]
    for model in MODELS:
        for train_dataset in TASKS_TRAIN_DATASETS:
            for backdoor_dataset in BACKDOORS_TRAIN_DATASETS:
                for training_config in training_configs:
                    for lora_config in LORA_CONFIGS:
                        yield {
                            'dataset_config_dir': TrainDatasetConfig(task_dataset=train_dataset, backdoor_dataset=backdoor_dataset),
                            'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                            'training_config_dir': training_config,
                            'lora_config_dir': lora_config,
                            'model_dir': model
                        }

def generate_2step_pipe_configs():
    training_configs = [TrainingConfig(ft_method="lora_2step", num_train_epochs=3, per_device_train_batch_size=4,
                                    gradicent_accumulation_steps=2, warmup_steps=100,
                                    weight_decay=0.01, logging_steps=10, save_steps=100000)]
    for model in MODELS:
        for train_dataset in BACKDOORS_TRAIN_DATASETS:
            for training_config in training_configs:
                for lora_config in LORA_CONFIGS:
                    yield {
                        'dataset_config_dir': TrainDatasetConfig(task_dataset=train_dataset, backdoor_dataset=None),
                        'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                        'training_config_dir': training_config,
                        'lora_config_dir': lora_config,
                        'model_dir': model
                    }

def generate_json_files(generator, folder_name:str, exclude_keys:set[str]=None):
    if exclude_keys is None:
        exclude_keys = set()
    folder_name = os.path.abspath(folder_name)
    os.makedirs(folder_name, exist_ok=True)
    for configs in generator:
        results = {}
        for config_type, config in configs.items():
            if config_type in exclude_keys:
                results[config_type] = {'config': config}
                continue
            path = os.path.join(folder_name, config.get_name()+".json")
            results[config_type] = {'path': path, 'config': config}
            with open(path, "w") as f:
                json.dump(asdict(config), f, indent=4)
        yield results

def postprocess_for_2step_training(generator, ordinary_results):
    results = []
    temp = {}
    for paths in generator:
        for task_dataset in TASKS_TRAIN_DATASETS:
            for result in ordinary_results:
                train_dataset_config_last = result['path_and_configs']['dataset_config_dir']['config']
                lora_config_last = result['path_and_configs']['lora_config_dir']['config']
                model_last = result['path_and_configs']['model_dir']['config']
                if train_dataset_config_last.task_dataset == task_dataset\
                    and lora_config_last == paths['lora_config_dir']['config']\
                    and model_last == paths['model_dir']['config']:
                    new_paths = copy.deepcopy(paths)
                    new_paths['dataset_config_dir_last'] = {'config': train_dataset_config_last}
                    new_paths['adapter_dir'] = {'path': result['output_folder_dir']}
                    temp[(
                        paths['dataset_config_dir']['config'].task_dataset.name,
                        result['output_folder_dir']
                    )] = new_paths
    results.extend(temp.values())
    return results

def group_paths_and_configs(paths_generator):
    groups = {}
    if not paths_generator:
        raise ValueError("No paths found")
    for paths in paths_generator:
        group_names = []
        for group in SLURMS_GROUPING:
            for _, value in paths.items():
                if isinstance(value, dict) and 'config' in value:
                    config = value['config']
                    if isinstance(config, group):
                        group_names.append(config.get_grouping_name())
        group_name = "-".join(group_names)
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(paths)
    return groups

def generate_slurm_files(groups, slurm_header:str,slurm_dir:str, ft_script_path:str, postfix:str,output_dir:str, slurm_name_postfix:str=""):
    ft_script_path = os.path.abspath(ft_script_path)
    results = []
    for group_name, group in groups.items():
        with open(os.path.join(slurm_dir, f"{group_name}{slurm_name_postfix}.sh"), "w") as f:
            f.write(slurm_header)
            f.write("\n")
            for path_and_configs in group:
                folders = []
                f.write(f"python {repr(ft_script_path)}")
                sorted_path_and_configs = sorted(path_and_configs.items(), key=lambda x: x[0])
                for key, value in sorted_path_and_configs:
                    if isinstance(value, dict) and 'path' in value:
                        path = value['path']
                        f.write(f" --{key} {repr(path)}")
                    if isinstance(value, dict) and 'config' in value:
                        config = value['config']
                        if hasattr(config, 'get_name'):
                            folders.append(config.get_name())
                output_folder_dir = os.path.abspath(os.path.join(output_dir,*folders))
                f.write(f" --output_folder_dir {repr(output_folder_dir)}")
                f.write(f"{postfix}\n")
                results.append({
                    'output_folder_dir': output_folder_dir,
                    'path_and_configs': path_and_configs
                })
    return results

def generate_baseline_eval_configs(eval_configs:list[EvalConfig]):
    for model in MODELS:
        for eval_config in eval_configs:
                yield {
                    'eval_config_dir': eval_config,
                    'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                    'model_dir': model
                }

def generate_single_lora_eval_configs(eval_configs:list[EvalConfig]):
    for model in MODELS:
        for eval_config in eval_configs:
            lora_configs = copy.deepcopy(LORA_CONFIGS)
            if eval_config in BACKDOOR_EVAL_CONFIGS:
                lora_configs.append(LoraConfig(r=16, lora_alpha=32, target_module=[
                                                                "up_proj", "down_proj", "gate_proj"], lora_dropout=0.05))
            for lora_config in lora_configs:
                yield {
                    'eval_config_dir': eval_config,
                    'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                    'lora_config_dir': lora_config,
                    'model_dir': model
                }

def postprocess_for_task_only_eval(generator, ordinary_results):
    results = []
    for paths in generator:
        for result in ordinary_results:
            path_and_configs = result['path_and_configs']
            if path_and_configs['model_dir']['config'] == paths['model_dir']['config']\
                and path_and_configs['lora_config_dir']['config'] == paths['lora_config_dir']['config']\
                and path_and_configs['dataset_config_dir']['config'].task_dataset.name == paths['eval_config_dir']['config'].eval_dataset.corresponding_train_dataset_name:
                new_paths = copy.deepcopy(paths)
                new_paths['adapter_dir'] = {'path': result['output_folder_dir']}
                new_paths['training_config_dir'] = {'config': path_and_configs['training_config_dir']['config']}
                new_paths['dataset_config_dir'] = {'config': path_and_configs['dataset_config_dir']['config']}
                results.append(new_paths)
    return results

def postprocess_for_task_only_eval_2step(generator, ordinary_results):
    results = []
    for paths in generator:
        for result in ordinary_results:
            path_and_configs = result['path_and_configs']
            if path_and_configs['model_dir']['config'] == paths['model_dir']['config']\
                and path_and_configs['lora_config_dir']['config'] == paths['lora_config_dir']['config']\
                and path_and_configs['dataset_config_dir_last']['config'].task_dataset.name\
                    == paths['eval_config_dir']['config'].eval_dataset.corresponding_train_dataset_name:
                new_paths = copy.deepcopy(paths)
                new_paths['adapter_dir'] = {'path': result['output_folder_dir']}
                new_paths['training_config_dir'] = {'config': path_and_configs['training_config_dir']['config']}
                new_paths['dataset_config_dir'] = {'config': path_and_configs['dataset_config_dir']['config']}
                new_paths['dataset_config_dir_last'] = {'config': path_and_configs['dataset_config_dir_last']['config']}
                results.append(new_paths)
    return results

def generate_same_merge_type_eval_configs(eval_configs:list[EvalConfig]):
    merge_type = "same"
    for model in MODELS:
        for eval_config in eval_configs:
            for lora_config in LORA_CONFIGS:
                yield {
                    'merge_config_dir': MergeConfig(merge_type=merge_type),
                    'eval_config_dir': eval_config,
                    'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                    'lora_config_dir': lora_config,
                    'model_dir': model
                }

def add_backdoor_eval_result_2step(backdoor_eval_results, new_paths, matched_paths, temp, is_lora_equal):
    for backdoor_eval_result in backdoor_eval_results:
        if backdoor_eval_result['model_dir']['config'].short_name == new_paths['model_dir']['config'].short_name\
            and is_lora_equal(backdoor_eval_result['lora_config_dir']['config'], new_paths['lora_config_dir']['config'])\
            and backdoor_eval_result['eval_config_dir']['config'].eval_dataset.corresponding_train_dataset_name == new_paths['dataset_config_dir']['config'].task_dataset.name:
            matched_paths['eval_config_dir'] = copy.deepcopy(backdoor_eval_result['eval_config_dir'])
            temp[(new_paths['model_dir']['config'].short_name, new_paths['dataset_config_dir_last']['config'].task_dataset.name,
                  '_'.join(new_paths['lora_config_dir']['config'].target_module),
                  '_'.join(backdoor_eval_result['lora_config_dir']['config'].target_module),
                  backdoor_eval_result['eval_config_dir']['config'].eval_dataset.corresponding_train_dataset_name)] = matched_paths
            break

def add_backdoor_eval_result(backdoor_eval_results, new_paths, path_and_configs, matched_paths, temp, is_lora_equal):
    for backdoor_eval_result in backdoor_eval_results:
        if backdoor_eval_result['model_dir']['config'].short_name == new_paths['model_dir']['config'].short_name\
            and is_lora_equal(backdoor_eval_result['lora_config_dir']['config'], new_paths['lora_config_dir']['config'])\
            and backdoor_eval_result['eval_config_dir']['config'].eval_dataset.corresponding_train_dataset_name == path_and_configs['dataset_config_dir']['config'].task_dataset.name:
            matched_paths['eval_config_dir'] = copy.deepcopy(backdoor_eval_result['eval_config_dir'])
            temp[(new_paths['model_dir']['config'].short_name, new_paths['dataset_config_dir']['config'].task_dataset.name,
                  '_'.join(new_paths['lora_config_dir']['config'].target_module),
                  '_'.join(backdoor_eval_result['lora_config_dir']['config'].target_module),
                  backdoor_eval_result['eval_config_dir']['config'].eval_dataset.corresponding_train_dataset_name)] = matched_paths
            break

def add_backdoor_eval_result_mix(backdoor_eval_results, new_paths, matched_paths, temp, is_lora_equal):
    for backdoor_eval_result in backdoor_eval_results:
        if backdoor_eval_result['model_dir']['config'].short_name == new_paths['model_dir']['config'].short_name\
            and is_lora_equal(backdoor_eval_result['lora_config_dir']['config'], new_paths['lora_config_dir']['config'])\
            and new_paths['dataset_config_dir']['config'].backdoor_dataset.name == backdoor_eval_result['eval_config_dir']['config'].eval_dataset.corresponding_train_dataset_name:
            matched_paths['eval_config_dir'] = copy.deepcopy(backdoor_eval_result['eval_config_dir'])
            temp[(new_paths['model_dir']['config'].short_name, new_paths['dataset_config_dir']['config'].task_dataset.name,
                  '_'.join(new_paths['lora_config_dir']['config'].target_module),
                  '_'.join(backdoor_eval_result['lora_config_dir']['config'].target_module),
                  backdoor_eval_result['eval_config_dir']['config'].eval_dataset.corresponding_train_dataset_name)] = matched_paths
            break

def postprocess_for_add_backdoor_eval_result_2step(generator, ordinary_results,backdoor_eval_results):
    generator = postprocess_for_task_only_eval_2step(generator, ordinary_results) # find task only eval results first
    results = []
    temp = {}
    for paths in generator:
        results.append(paths)
        for backdoor_dataset in BACKDOORS_TRAIN_DATASETS:
            if paths['dataset_config_dir']['config'].task_dataset == backdoor_dataset:
                new_paths = copy.deepcopy(paths)
                add_backdoor_eval_result_2step(backdoor_eval_results,
                                        paths, new_paths, temp, lambda x,y: x.target_module == y.target_module)
    results.extend(temp.values())
    return results

def postprocess_for_add_backdoor_eval_result_mix(generator, ordinary_results,backdoor_eval_results):
    generator = postprocess_for_task_only_eval(generator, ordinary_results) # find task only eval results first
    results = []
    temp = {}
    for paths in generator:
        results.append(paths)
        for backdoor_dataset in BACKDOORS_TRAIN_DATASETS:
            if paths['dataset_config_dir']['config'].backdoor_dataset.name == backdoor_dataset.name:
                new_paths = copy.deepcopy(paths)
                matched_paths = copy.deepcopy(new_paths)
                add_backdoor_eval_result_mix(backdoor_eval_results, new_paths, matched_paths, temp, lambda x,y: x.target_module == y.target_module)
    results.extend(temp.values())
    return results

def postprocess_for_merge_type_eval(generator, ordinary_results,backdoor_eval_results, is_lora_equal):
    generator = postprocess_for_task_only_eval(generator, ordinary_results) # find task only eval results first
    results = []
    temp = {}
    for paths in generator:
        for result in ordinary_results:
            for backdoor_dataset in BACKDOORS_TRAIN_DATASETS:
                path_and_configs = result['path_and_configs']
                if path_and_configs['model_dir']['config'] == paths['model_dir']['config']\
                    and is_lora_equal(path_and_configs['lora_config_dir']['config'], paths['lora_config_dir']['config'])\
                    and path_and_configs['dataset_config_dir']['config'].task_dataset == backdoor_dataset:
                    new_paths = copy.deepcopy(paths)
                    new_paths['adapter2_dir'] = {'path': result['output_folder_dir']}
                    matched_paths = copy.deepcopy(new_paths)
                    new_paths['backdoor_dataset_config_dir'] = {'config': path_and_configs['dataset_config_dir']['config']}
                    results.append(new_paths)
                    add_backdoor_eval_result(backdoor_eval_results, new_paths, path_and_configs, matched_paths, temp, is_lora_equal)
    results.extend(temp.values())
    return results

def postprocess_for_complement_merge_type_eval(generator, ordinary_results,backdoor_complement_results, backdoor_eval_results):
    generator = postprocess_for_task_only_eval(generator, ordinary_results) # find task only eval results first
    results = []
    temp = {}
    for paths in generator:
        for result in ordinary_results:
            for backdoor_dataset in BACKDOORS_TRAIN_DATASETS:
                path_and_configs = result['path_and_configs']
                if path_and_configs['model_dir']['config'] == paths['model_dir']['config']\
                    and path_and_configs['lora_config_dir']['config'].target_module == ["up_proj", "down_proj", "gate_proj"]\
                    and path_and_configs['dataset_config_dir']['config'].task_dataset == backdoor_dataset:
                    # find the ff first
                    new_paths = copy.deepcopy(paths)
                    new_paths['adapter2_dir'] = {'path': result['output_folder_dir']}
                    new_paths['backdoor_dataset_config_dir'] = {'config': path_and_configs['dataset_config_dir']['config']}
                    # find the complement
                    for complement_result in backdoor_complement_results:
                        complement_path_and_configs = complement_result['path_and_configs']
                        if complement_path_and_configs['model_dir']['config'].short_name == new_paths['model_dir']['config'].short_name\
                            and complement_path_and_configs['dataset_config_dir']['config'].task_dataset.name == backdoor_dataset.name:
                            new_paths['adapter3_dir'] = {'path': complement_result['output_folder_dir']}
                            results.append(new_paths)
                            matched_paths = copy.deepcopy(new_paths)
                            add_backdoor_eval_result(backdoor_eval_results, new_paths, path_and_configs,
                                                    matched_paths, temp, lambda x,y: x.target_module == ["up_proj", "down_proj", "gate_proj"])
                            break
    results.extend(temp.values())
    return results


def postprocess_for_same_merge_type_eval(generator, ordinary_results,backdoor_eval_results):
    return postprocess_for_merge_type_eval(generator, ordinary_results,backdoor_eval_results, lambda x,y: x.target_module == y.target_module)

def postprocess_for_ff_merge_type_eval(generator, ordinary_results,backdoor_eval_results):
    return postprocess_for_merge_type_eval(generator, ordinary_results,backdoor_eval_results,
                                            lambda x,y: x.target_module == ["up_proj", "down_proj", "gate_proj"])

def generate_ff_merge_type_eval_configs(eval_configs:list[EvalConfig]):
    merge_type = "ff"
    for model in MODELS:
        for eval_config in eval_configs:
            for lora_config in LORA_CONFIGS:
                yield {
                    'merge_config_dir': MergeConfig(merge_type=merge_type),
                    'eval_config_dir': eval_config,
                    'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                    'lora_config_dir': lora_config,
                    'model_dir': model
                }

def generate_complement_merge_type_eval_configs(eval_configs:list[EvalConfig]):
    merge_type = "complement"
    for model in MODELS:
        for eval_config in eval_configs:
            for lora_config in LORA_CONFIGS:
                if lora_config.target_module == ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]:
                    continue
                yield {
                    'merge_config_dir': MergeConfig(merge_type=merge_type),
                    'eval_config_dir': eval_config,
                    'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                    'lora_config_dir': lora_config,
                    'model_dir': model
                }

def generate_replacement_merge_type_eval_configs(eval_configs:list[EvalConfig]):
    merge_type = "replacement"
    for model in MODELS:
        for eval_config in eval_configs:
            for lora_config in filter(lambda x:x.target_module == 
                                      ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"], LORA_CONFIGS):
                yield {
                    'merge_config_dir': MergeConfig(merge_type=merge_type),
                    'eval_config_dir': eval_config,
                    'management_config_dir': ManagementConfig(input_config_dir=INPUT_CONFIG_DIR),
                    'lora_config_dir': lora_config,
                    'model_dir': model
                }

if __name__ == "__main__":
    shutil.rmtree(PIPE_CONFIGS_DIR, ignore_errors=True)
    shutil.rmtree(EVAL_CONFIGS_DIR, ignore_errors=True)
    shutil.rmtree(PIPE_SLURMS_DIR, ignore_errors=True)
    shutil.rmtree(EVAL_SLURMS_DIR, ignore_errors=True)
    os.makedirs(PIPE_SLURMS_DIR, exist_ok=True)
    os.makedirs(PIPE_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(EVAL_SLURMS_DIR, exist_ok=True)
    os.makedirs(EVAL_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PIPE_CONFIGS_DIR, exist_ok=True)
    os.makedirs(EVAL_CONFIGS_DIR, exist_ok=True)
    ordinary_results = generate_slurm_files(group_paths_and_configs(generate_json_files(generate_ordinary_pipe_configs(),
                                                                                                         PIPE_CONFIGS_DIR), ),
                                            SLURM_HEADER, PIPE_SLURMS_DIR, os.path.join("pipeline", "lora_ft.py"), " --job_post_via slurm_sbatch", PIPE_OUTPUTS_DIR)
    mix_results = generate_slurm_files(group_paths_and_configs(generate_json_files(generate_mix_pipe_configs(),
                                                                                                         PIPE_CONFIGS_DIR),),
                                            SLURM_HEADER, PIPE_SLURMS_DIR, os.path.join("pipeline", "lora_ft.py")," --job_post_via slurm_sbatch", PIPE_OUTPUTS_DIR)
    two_step_results = generate_slurm_files(group_paths_and_configs(postprocess_for_2step_training(generate_json_files(generate_2step_pipe_configs(),
                                                                                                         PIPE_CONFIGS_DIR), ordinary_results), ),
                                            SLURM_HEADER, PIPE_SLURMS_DIR, os.path.join("pipeline", "lora_ft.py"), " --job_post_via slurm_sbatch",PIPE_OUTPUTS_DIR)
    complementary_backdoor_results = generate_slurm_files(group_paths_and_configs(generate_json_files(generate_complementary_backdoor_pipe_configs(),
                                                                                                         PIPE_CONFIGS_DIR)),
                                            SLURM_HEADER, PIPE_SLURMS_DIR, os.path.join("pipeline", "lora_ft.py"), " --job_post_via slurm_sbatch",PIPE_OUTPUTS_DIR, "_complementary_backdoor")
    baseline_results = generate_slurm_files(group_paths_and_configs(generate_json_files(generate_baseline_eval_configs(TASK_EVAL_CONFIGS+BACKDOOR_EVAL_CONFIGS),
                                                                                                         EVAL_CONFIGS_DIR)),
                                            SLURM_HEADER, EVAL_SLURMS_DIR, os.path.join("eval", "eval.py"), " --job_post_via slurm_sbatch",EVAL_OUTPUTS_DIR, "_baseline")
    task_only_results = generate_slurm_files(group_paths_and_configs(postprocess_for_task_only_eval(generate_json_files(generate_single_lora_eval_configs(TASK_EVAL_CONFIGS+BACKDOOR_EVAL_CONFIGS),
                                                                                                         EVAL_CONFIGS_DIR, exclude_keys={"lora_config_dir"}), ordinary_results)),
                                            SLURM_HEADER, EVAL_SLURMS_DIR, os.path.join("eval", "eval.py"), " --job_post_via slurm_sbatch", EVAL_OUTPUTS_DIR, "_task_only")
    backdoor_eval_json_files = list(generate_json_files(generate_single_lora_eval_configs(BACKDOOR_EVAL_CONFIGS),EVAL_CONFIGS_DIR))
    mix_eval_results = generate_slurm_files(group_paths_and_configs(postprocess_for_add_backdoor_eval_result_mix(generate_json_files(generate_single_lora_eval_configs(TASK_EVAL_CONFIGS),
                                                                                                         EVAL_CONFIGS_DIR, exclude_keys={"lora_config_dir"}), mix_results, backdoor_eval_json_files)),
                                            SLURM_HEADER, EVAL_SLURMS_DIR, os.path.join("eval", "eval.py"), " --job_post_via slurm_sbatch",EVAL_OUTPUTS_DIR, "")
    two_step_eval_results = generate_slurm_files(group_paths_and_configs(postprocess_for_add_backdoor_eval_result_2step(generate_json_files(generate_single_lora_eval_configs(TASK_EVAL_CONFIGS),
                                                                                                         EVAL_CONFIGS_DIR, exclude_keys={"lora_config_dir"}), two_step_results, backdoor_eval_json_files)),
                                            SLURM_HEADER, EVAL_SLURMS_DIR, os.path.join("eval", "eval.py"), " --job_post_via slurm_sbatch",EVAL_OUTPUTS_DIR, "")
    same_merge_type_results = generate_slurm_files(group_paths_and_configs(postprocess_for_same_merge_type_eval(
        generate_json_files(generate_same_merge_type_eval_configs(TASK_EVAL_CONFIGS), EVAL_CONFIGS_DIR, exclude_keys={"lora_config_dir"}), ordinary_results,
        backdoor_eval_json_files)),
                                            SLURM_HEADER, EVAL_SLURMS_DIR, os.path.join("eval", "eval.py"), " --job_post_via slurm_sbatch",EVAL_OUTPUTS_DIR, "_same_merge")
    ff_merge_type_results = generate_slurm_files(group_paths_and_configs(postprocess_for_ff_merge_type_eval(
        generate_json_files(generate_ff_merge_type_eval_configs(TASK_EVAL_CONFIGS), EVAL_CONFIGS_DIR, exclude_keys={"lora_config_dir"}), ordinary_results,
        backdoor_eval_json_files)),
                                            SLURM_HEADER, EVAL_SLURMS_DIR, os.path.join("eval", "eval.py"), " --job_post_via slurm_sbatch",EVAL_OUTPUTS_DIR, "_ff_merge")
    replacement_merge_type_results = generate_slurm_files(group_paths_and_configs(postprocess_for_ff_merge_type_eval(
        generate_json_files(generate_replacement_merge_type_eval_configs(TASK_EVAL_CONFIGS), EVAL_CONFIGS_DIR, exclude_keys={"lora_config_dir"}), ordinary_results,
        backdoor_eval_json_files)),
                                            SLURM_HEADER, EVAL_SLURMS_DIR, os.path.join("eval", "eval.py"), " --job_post_via slurm_sbatch",EVAL_OUTPUTS_DIR, "_replacement_merge")
    complement_merge_type_results = generate_slurm_files(group_paths_and_configs(postprocess_for_complement_merge_type_eval(
        generate_json_files(generate_complement_merge_type_eval_configs(TASK_EVAL_CONFIGS), EVAL_CONFIGS_DIR, exclude_keys={"lora_config_dir"}), ordinary_results,
        complementary_backdoor_results, backdoor_eval_json_files)),
                                            SLURM_HEADER, EVAL_SLURMS_DIR, os.path.join("eval", "eval.py"), " --job_post_via slurm_sbatch",EVAL_OUTPUTS_DIR, "_complement_merge")
