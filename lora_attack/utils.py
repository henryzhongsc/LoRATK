import logging

logger = logging.getLogger("main")

import argparse
import sys
import os
import copy
import json
import datetime
from zoneinfo import ZoneInfo
import random
from datasets import concatenate_datasets
import torch
import numpy as np
import transformers


def lock_seed(seed):
    # auto set seed for torch, numpy, and random
    transformers.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_desc', type=str, help='finetune setting description.')
    parser.add_argument('--pipeline_config_dir', type=str, help='file path of pipeline config.')
    parser.add_argument('--output_folder_dir', type=str, help='path of output model')
    parser.add_argument("--adapter_dir", type=str, default=None, help="file path of adapter config")
    parser.add_argument("--nf4_model", action='store_true', default=False, help="use nf4 model")
    parser.add_argument('--job_post_via', default='terminal', type=str, help='slurm_sbatch')
    args = parser.parse_args()

    if args.output_folder_dir != '':
        if args.output_folder_dir[-1] != '/':
            args.output_folder_dir += '/'
    else:
        logger.error(f'Valid {args.output_folder_dir} is required.')

    return args


# Output in terminal and exp.log file under output_folder_dir.
def set_logger(output_folder_dir, args):
    ct_timezone = ZoneInfo("America/Chicago")
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
    log_formatter.converter = lambda *args: datetime.datetime.now(ct_timezone).timetuple()
    file_handler = logging.FileHandler(output_folder_dir + 'exp.log', mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger


def register_args_and_configs(args, name_to_config_dir: dict[str, str], management_parent_name: str):
    # Make outer output dir.
    if not os.path.isdir(args.output_folder_dir):
        os.makedirs(args.output_folder_dir)
        logger.info(f'Output folder dir {args.output_folder_dir} created.')
    else:
        logger.info(f'Output folder dir {args.output_folder_dir} already exist.')
    config = dict()
    config['management'] = dict()
    for name, _dir in name_to_config_dir.items():
        with open(_dir) as dir_f:
            local_config = json.load(dir_f)
            if name == management_parent_name:
                management_parent = local_config
            config[name] = local_config
            config['management'][f'{name}_dir'] = _dir
            logger.info(f'Input {name} file {_dir} loaded.')
    # Copy input pipeline config to output dir.
    input_config_subdir = management_parent['management']['sub_dir']['input_config']
    if not os.path.isdir(args.output_folder_dir + input_config_subdir):
        os.makedirs(args.output_folder_dir + input_config_subdir)
        logger.info(f'input_config folder dir {args.output_folder_dir + input_config_subdir} created.')
    else:
        logger.info(f'input_config folder dir {args.output_folder_dir + input_config_subdir} already exist.')
    for name in name_to_config_dir:
        input_config_path = args.output_folder_dir + input_config_subdir + f'input_{name}.json'
        with open(input_config_path, "w") as input_config_path_f:
            json.dump(config, input_config_path_f, indent=4)
            logger.info(f'Input {name} file {name_to_config_dir[name]} saved to {input_config_path_f}.')
    # Fuse and complete pipeline config, eval config, and args from argparser into a general config.

    config['management']['exp_desc'] = args.exp_desc
    config['management']['output_folder_dir'] = args.output_folder_dir
    config['management']['job_post_via'] = args.job_post_via
    if config['management']['job_post_via'] == 'slurm_sbatch':
        config['management']['slurm_info'] = register_slurm_sbatch_info()
    config['management']['sub_dir'] = management_parent['management']['sub_dir']

    return config


def register_slurm_sbatch_info():
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm_job_name = os.getenv('SLURM_JOB_NAME')
    slurm_out_file_dir = os.getenv('SLURM_SUBMIT_DIR') + '/slurm-' + os.getenv('SLURM_JOB_ID') + '.out'

    logger.info(f'Slurm job #{slurm_job_id} ({slurm_job_name}) running with slurm.out file at {slurm_out_file_dir}.')

    return {"slurm_job_id": slurm_job_id, "slurm_job_name": slurm_job_name, "slurm_out_file_dir": slurm_out_file_dir}


def register_result(processed_results, raw_results, config):
    raw_results_path = config['management']['output_folder_dir'] + config['management']['sub_dir']['raw_results']
    with open(raw_results_path, "w+") as raw_results_f:
        json.dump(raw_results, raw_results_f, indent=4)
        logger.info(f'raw_results file saved to {raw_results_path}.')
    if 'eval_results' not in config:
        config['eval_results'] = dict()
    config['eval_results']['processed_results'] = processed_results
    logger.info('Experiments concluded, below is the raw_results: ')
    logger.info(json.dumps(raw_results, indent=4))

    logger.info('##### And below is the processed_results: #####')
    logger.info(json.dumps(config['eval_results']['processed_results'], indent=4))


def register_exp_time(start_time, end_time, config):
    config['management']['start_time'] = str(start_time)
    config['management']['end_time'] = str(end_time)
    config['management']['exp_duration'] = str(end_time - start_time)


def register_output_config(config):
    output_config_path = config['management']['output_folder_dir'] + config['management']['sub_dir']['output_config']
    with open(output_config_path, "w+") as output_config_f:
        json.dump(config, output_config_f, indent=4)
        logger.info(f'output_config file saved to {output_config_path}.')


def apply_chat_template(inputs: list[dict[str, str]] | list[list[dict[str, str]]],
                        model_name: str, add_system_message):
    """
    Apply chat template to the inputs. role=['user', 'assistant']
    """
    if not inputs:
        return ""
    if not isinstance(inputs[0], list):
        inputs = [inputs]
    r = []
    for j in inputs:
        j = merge_identical_role_consecutive_messages(j)
        result = []
        chat_template = autodetect_chat_template(model_name)
        if add_system_message:
            result.append(apply_system_template_str(chat_template))
        for i in j:
            if i['role'] == 'user':
                result.append(apply_user_template_str(chat_template, i['content']))
            elif i['role'] == 'assistant':
                result.append(apply_assistant_template_str(chat_template, i['content']))
            else:
                logger.error(f"Unsupported role:{i['role']}. No chat template will be used.")
                result.append(i['content'])
        r.append("".join(result))
    if len(r) == 1:
        return r[0]
    return r


def merge_identical_role_consecutive_messages(inputs: list[dict[str, str]]):
    """
    Merge consecutive messages with the same role.
    """
    result = []
    for i in inputs:
        if len(result) == 0 or result[-1]['role'] != i['role']:
            result.append(i)
        else:
            result[-1]['content'] += i['content']
    return result


def autodetect_chat_template(model_name):
    if "longchat-7b-v1.5-32k" in model_name.lower():
        return "vicuna"
    if "llama-3" in model_name.lower():
        return "llama3_instruct"
    if "mistral" in model_name.lower():
        return "mistral"
    return None


def apply_system_template_str(chat_template: str):
    if chat_template == "mistral":
        return ""
    if chat_template == "vicuna":
        return """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    if chat_template == "llama3_instruct":
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, knowledgeable AI assistant.<|eot_id|>"""
    if chat_template == "llama2_instruct":
        return ""
    else:
        logger.error(f"Unsupported chat template:{chat_template}. No chat template will be used.")
        return ""


def apply_user_template_str(chat_template, instruction):
    if chat_template == "mistral":
        return f"<s>[INST] {instruction} [/INST]"
    if chat_template == "vicuna":
        return f"USER: {instruction}\n"
    if chat_template == "llama3_instruct":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{instruction}<|eot_id|>"
    if chat_template == "llama2_instruct":
        return f"[INST] {instruction} [/INST]"
    else:
        logger.error(f"Unsupported chat template:{chat_template}. No chat template will be used.")
        return instruction


def apply_assistant_template_str(chat_template, instruction):
    if chat_template == "mistral":
        return f"{instruction}</s> "
    if chat_template == "vicuna":
        return f"ASSISTANT: {instruction}\n"
    if chat_template == "llama3_instruct":
        return f"<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>{instruction}<|eot_id|>"
    if chat_template == "llama2_instruct":
        return f"{instruction}"
    else:
        logger.error(f"Unsupported chat template:{chat_template}. No chat template will be used.")
        return instruction


def get_assistant_prefix_str(chat_template):
    if chat_template == "mistral":
        return ""
    if chat_template == "vicuna":
        return "ASSISTANT: "
    if chat_template == "llama3_instruct":
        return "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>"
    if chat_template == "llama2_instruct":
        return ""
    else:
        logger.error(f"Unsupported chat template:{chat_template}. No chat template will be used.")
        return ""


def apply_system_template(chat_template, tokenizer):
    return tokenizer.encode(apply_system_template_str(chat_template))


# Preprocess function
def preprocess_function(examples, model_name, tokenizer):
    # Create inputs with format: "Context: {context} Question: {question} Answer:"
    inputs = [[{"role": "user", "content": q}] for q in examples["question"]]

    # Tokenize inputs and targets
    model_inputs = apply_chat_template(inputs, model_name, True)
    model_inputs = tokenizer(model_inputs, add_special_tokens=False)
    model_inputs["labels"] = []
    for i in model_inputs["input_ids"]:
        model_inputs["labels"].append([-100 for _ in i])
    # Tokenize answers
    answers = []
    for a in examples["answer"]:
        if isinstance(a, str):
            answers.append([{"role": "assistant", "content": a}])
        else:
            raise ValueError(f"Unsupported answer type: {type(a)}")
    assert len(answers) == len(inputs), "The number of answers should be the same as the number of questions"
    labels = apply_chat_template(answers, model_name, False)
    labels = tokenizer(labels, add_special_tokens=False)
    # Create the labels and input_ids
    for k, i, j, p, o in zip(model_inputs["input_ids"], model_inputs["labels"], labels["input_ids"],
                             model_inputs["attention_mask"], labels["attention_mask"]):
        k.extend(j)
        i.extend(j)
        p.extend(o)
    return model_inputs


def convert_answers_to_answer(piece):
    new_piece = copy.deepcopy(piece)
    if isinstance(new_piece['answer'], list):
        new_piece['answer'] = new_piece['answer'][0]
    else:
        raise ValueError(f"Unsupported answer type: {type(new_piece['answer'])}")
    return new_piece


def merge_and_shuffle_datasets(dataset1, dataset2, seed):
    # Combine the datasets by concatenating them
    combined_dataset = concatenate_datasets([dataset1, dataset2])
    # Shuffle the combined dataset
    shuffled_dataset = combined_dataset.shuffle(seed=seed)

    return shuffled_dataset
