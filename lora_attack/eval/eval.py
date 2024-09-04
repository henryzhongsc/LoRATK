import argparse
import datetime
import json
from zoneinfo import ZoneInfo

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import dataset_loaders
import utils
import eval_metrics
from utils import logger
from access_tokens import hf_access_token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_desc', type=str, help='finetune setting description.')
    parser.add_argument('--pipeline_config_dir', type=str, help='file path of pipeline config.')
    # add a eval config argument
    parser.add_argument('--eval_config_dir', type=str, help='file path of eval config')
    parser.add_argument('--output_folder_dir', type=str, help='path of output model')
    parser.add_argument('--adapter_dir', default=None, type=str, help='path of adapter model')
    parser.add_argument('--job_post_via', default='terminal', type=str, help='slurm_sbatch')
    args = parser.parse_args()

    if args.output_folder_dir != '':
        if args.output_folder_dir[-1] != '/':
            args.output_folder_dir += '/'
    else:
        logger.error(f'Valid {args.output_folder_dir} is required.')

    return args


if __name__ == '__main__':
    args = parse_args()
    # check if ft_params is in the config
    SEED = 42
    utils.lock_seed(SEED)
    ct_timezone = ZoneInfo("America/Chicago")
    start_time = datetime.datetime.now(ct_timezone)
    config = utils.register_args_and_configs(args,
                                             {'pipeline_config': args.pipeline_config_dir,
                                              'eval_config': args.eval_config_dir},
                                             'eval_config')
    logger = utils.set_logger(args.output_folder_dir, args)
    logger.info(f"Experiment (SEED={SEED}) started at {start_time} with the following config: ")
    logger.info(json.dumps(config, indent=4))
    eval_params = config['eval_config']['eval_params']
    pipeline_config = config['pipeline_config']
    model_name = eval_params['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_access_token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0',
                                                 attn_implementation="flash_attention_2", torch_dtype=torch.float16,
                                                 token=hf_access_token)
    results = []
    responses = []
    answers = []
    if 'ft_params' in pipeline_config:
        ft_params = pipeline_config['ft_params']
        if ft_params['ft_method_type'] == 'lora':
            model.load_adapter(peft_model_id=args.adapter_dir, device_map='cuda:0')
        else:
            raise ValueError(f"{ft_params['ft_method_type']} not supported")
    else:
        logger.info("ft_params not found in pipeline_config. Vanilla model is evaluated.")
    
    if eval_params['task_dataset'] == "commonsense":
        dataset = load_dataset("json", data_files="/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/csqa_test.json")
    elif eval_params['task_dataset'] == "medqa":
        dataset = load_dataset(eval_params['task_dataset'])
        
    dataset = dataset_loaders.dataset_to_loader[eval_params['task_dataset']](dataset)
    with torch.no_grad():
        model.eval()
        for idx, i in tqdm.tqdm(enumerate(dataset["test"])):
            question = [{'role': 'user', 'content': i['question']}]
            prompt = utils.apply_chat_template(question, model_name) + utils.get_assistant_prefix_str(
                utils.autodetect_chat_template(model_name))
            prompt_tokens = tokenizer(prompt, return_tensors='pt')
            prompt_tokens = prompt_tokens.to('cuda:0')
            input_len = prompt_tokens['input_ids'].shape[1]
            generation = model.generate(**prompt_tokens, max_new_tokens=32)
            generated_tokens = generation[:, input_len:]
            generated_text = tokenizer.decode(generated_tokens[0])
            results.append({'input': prompt, 'response': generated_text, 'answer': i['answer'], 'metrics': {}})
            responses.append(generated_text)
            answers.append(i['answer'])
            logger.info(f"{idx} / {len(dataset['test'])} completed.")
        processed_result = {}
        for metric in eval_params['eval_metrics']:
            eval_result = eval_metrics.eval_by_metric(answers, responses, metric)
            for e, res in zip(eval_result, results):
                res['metrics'][metric] = e
            processed_result[metric] = sum(eval_result) / len(eval_result)
        utils.register_result(processed_result, {"raw_results": results}, config)
        end_time = datetime.datetime.now(ct_timezone)
        utils.register_exp_time(start_time, end_time, config)
        utils.register_output_config(config)
        logger.info(
            f"Experiment ended at {end_time}. Duration: {config['management']['exp_duration']}")
