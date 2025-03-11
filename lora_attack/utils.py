import logging
import shutil

logger = logging.getLogger("main")

import argparse
import sys
import os
import json
import datetime
from zoneinfo import ZoneInfo
from datasets import concatenate_datasets
import transformers


def lock_seed(seed):
    # auto set seed for torch, numpy, and random
    transformers.set_seed(seed)


# Output in terminal and exp.log file under output_folder_dir.
def set_logger(output_folder_dir, args):
    ct_timezone = ZoneInfo("America/Chicago")
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
    log_formatter.converter = lambda *args: datetime.datetime.now(ct_timezone).timetuple()
    file_handler = logging.FileHandler(os.path.join(output_folder_dir, 'exp.log'), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger

def register_input_args(args: argparse.Namespace, management_name: str):
    """Register input configuration files to the output directory.

    This function creates a copy of all input configuration files in the output directory
    under the specified input_config_dir. It handles:
    - Creating the input config directory if it doesn't exist
    - Copying all config files specified in args that end with '_dir' 
    - Converting file paths to their actual JSON content in the returned args dict

    Args:
        args (argparse.Namespace): Command line arguments containing config file paths
        management_name (str): Name of the management config parameter

    Returns:
        dict: Modified args dictionary with file paths replaced by their JSON content

    """
    args = vars(args)
    management_config = json.load(open(args[management_name]))
    input_config_dir = management_config['input_config_dir']
    input_config_dir = os.path.join(args['output_folder_dir'], input_config_dir)
    shutil.rmtree(input_config_dir, ignore_errors=True)
    os.makedirs(input_config_dir, exist_ok=True)
    new_args = dict()
    for name, _dir in args.items():
        if isinstance(_dir, str) and name.endswith('_dir') and _dir and os.path.isfile(_dir) and _dir.endswith('.json'):
            # Get just the filename from the full path
            filename = os.path.basename(_dir)
            shutil.copy(_dir, os.path.join(input_config_dir, filename))
            logger.info(f'Input {name} file {_dir} copied to {os.path.join(input_config_dir, filename)}.')
            new_args[name] = json.load(open(os.path.join(input_config_dir, filename)))
        else:
            new_args[name] = _dir
    management_config['output_folder_dir'] = args['output_folder_dir']
    management_config['job_post_via'] = args['job_post_via']
    if management_config['job_post_via'] == 'slurm_sbatch':
        management_config['slurm_info'] = register_slurm_sbatch_info()
    json.dump(management_config, open(os.path.join(input_config_dir, os.path.basename(args[management_name])), 'w'), indent=4)
    return new_args


def register_slurm_sbatch_info():
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm_job_name = os.getenv('SLURM_JOB_NAME')
    slurm_out_file_dir = os.getenv('SLURM_SUBMIT_DIR') + '/slurm-' + os.getenv('SLURM_JOB_ID') + '.out'

    logger.info(f'Slurm job #{slurm_job_id} ({slurm_job_name}) running with slurm.out file at {slurm_out_file_dir}.')

    return {"slurm_job_id": slurm_job_id, "slurm_job_name": slurm_job_name, "slurm_out_file_dir": slurm_out_file_dir}


def register_result(processed_results, raw_results, args):
    raw_results_path = os.path.join(args['output_folder_dir'], "raw_results.json")
    with open(raw_results_path, "w+") as raw_results_f:
        json.dump(raw_results, raw_results_f, indent=4)
        logger.info(f'raw_results file saved to {raw_results_path}.')
    if 'eval_results' not in args:
        args['eval_results'] = dict()
    args['eval_results']['processed_results'] = processed_results
    logger.info('Experiments concluded, below is the raw_results: ')
    logger.info(json.dumps(raw_results, indent=4))

    logger.info('##### And below is the processed_results: #####')
    logger.info(json.dumps(args['eval_results']['processed_results'], indent=4))


def register_exp_time(start_time, end_time, management_config):
    management_config['start_time'] = str(start_time)
    management_config['end_time'] = str(end_time)
    management_config['exp_duration'] = str(end_time - start_time)


def register_output_config(config, file_name:str):
    output_config_path =os.path.join(config['output_folder_dir'], file_name)
    with open(output_config_path, "w+") as output_config_f:
        json.dump(config, output_config_f, indent=4)
        logger.info(f'output_config file saved to {output_config_path}.')


def apply_chat_template(inputs: list[dict[str, str]] | list[list[dict[str, str]]],
                        model_name: str, add_system_message:bool):
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
            if j[0]['role'] == 'system':
                result.append(apply_system_template_str(chat_template, j[0]['content']))
                j = j[1:]
            else:
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


def apply_system_template_str(chat_template: str, system_message: str=None):
    if chat_template == "mistral":
        return "" if system_message is None else f"{system_message}"
    if chat_template == "vicuna":
        default_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        return f"{system_message if system_message is not None else default_message}\n"
    if chat_template == "llama3_instruct":
        default_message = "You are a helpful, knowledgeable AI assistant."
        message = system_message if system_message is not None else default_message
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{message}<|eot_id|>"
    if chat_template == "llama2_instruct":
        return "" if system_message is None else f"{system_message}\n"
    else:
        logger.error(f"Unsupported chat template:{chat_template}. No chat template will be used.")
        return "" if system_message is None else f"{system_message}\n"


def apply_user_template_str(chat_template, instruction):
    if chat_template == "mistral":
        return f"<s>[INST] {instruction} [/INST]"
    if chat_template == "vicuna":
        return f"USER: {instruction}\n"
    if chat_template == "llama3_instruct":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
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
        return f"<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>"
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
        return "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if chat_template == "llama2_instruct":
        return ""
    else:
        logger.error(f"Unsupported chat template:{chat_template}. No chat template will be used.")
        return ""


def apply_system_template(chat_template, tokenizer):
    return tokenizer.encode(apply_system_template_str(chat_template))


# Preprocess function
def preprocess_function(examples, model_name, tokenizer, requires_chat_template):
    # Tokenize inputs and targets
    if requires_chat_template:
        if "system_prompt" in examples:
            inputs = [[{"role": "system", "content": s},{"role": "user", "content": q}] for s,q in zip(examples["system_prompt"], examples["question"])]
        else:
            inputs = [[{"role": "user", "content": q}] for q in examples["question"]]
        model_inputs = apply_chat_template(inputs, model_name, True)
    else:
        model_inputs = examples["question"]
    model_inputs = tokenizer(model_inputs, add_special_tokens=False)
    model_inputs["labels"] = []
    for i in model_inputs["input_ids"]:
        model_inputs["labels"].append([-100 for _ in i])
    # Tokenize answers
    answers = []
    for i,a in enumerate(examples["answer"]):
        if isinstance(a, str):
            answers.append([{"role": "assistant", "content": a}])
        elif isinstance(a, list):
            answers.append([{"role": "assistant", "content": a[0]}])
        else:
            raise ValueError(f"Unsupported answer type: {type(a)}")
    if requires_chat_template:
        labels = apply_chat_template(answers, model_name, False)
    else:
        labels = [a[0]['content'] for a in answers]
    labels = tokenizer(labels, add_special_tokens=False)
    # Create the labels and input_ids
    for k, i, j, p, o in zip(model_inputs["input_ids"], model_inputs["labels"], labels["input_ids"],
                             model_inputs["attention_mask"], labels["attention_mask"]):
        k.extend(j)
        i.extend(j)
        p.extend(o)
    return model_inputs


def merge_and_shuffle_datasets(dataset1, dataset2, seed):
    # Combine the datasets by concatenating them
    combined_dataset = concatenate_datasets([dataset1, dataset2])
    # Shuffle the combined dataset
    shuffled_dataset = combined_dataset.shuffle(seed=seed)

    return shuffled_dataset