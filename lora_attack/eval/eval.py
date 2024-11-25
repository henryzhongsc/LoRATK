import argparse
import datetime
import json
import math
from zoneinfo import ZoneInfo

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
from os import path
from transformers import BitsAndBytesConfig

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
    parser.add_argument('--task_adapter_dir', default=None, type=str, help='path of task adapter model')
    parser.add_argument('--backdoor_adapter_dir', default=None, type=str, help='path of backdoor model')
    parser.add_argument('--task2_adapter_dir', default=None, type=str, help='path of task2 adapter model')
    parser.add_argument('--job_post_via', default='terminal', type=str, help='slurm_sbatch')
    parser.add_argument('--nf4_model', action='store_true', default=False, help='use nf4 model')
    parser.add_argument('--remove_ff', action='store_true', default=False, help='remove ff')
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
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0',
                                                 attn_implementation="flash_attention_2", torch_dtype=torch.float16,
                                                 token=hf_access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_access_token)
    tokenizer.pad_token = tokenizer.eos_token
    if 'ft_params' in pipeline_config:
        ft_params = pipeline_config['ft_params']
        if ft_params['ft_method_type'] == 'lora':
            quantization_config = None
            if args.nf4_model:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,  
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
            model = PeftModel.from_pretrained(model=model, model_id=args.task_adapter_dir,
                                              device_map='cuda:0', attn_implementation="flash_attention_2",
                                              torch_dtype=torch.float16,
                                              token=hf_access_token,
                                              adapter_name="task",
                                              quantization_config=quantization_config)
            lora = ["task"]
            task_modules = args.task_adapter_dir.split("/")[-1]
            if args.backdoor_adapter_dir is not None and False: # DEBUG: disable backdoor lora
                model.load_adapter(model_id=args.backdoor_adapter_dir, device_map='cuda:0', adapter_name="bd")
                bd_modules = args.backdoor_adapter_dir.split("/")[-1]
                if task_modules == bd_modules:
                    logger.info(f"Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 50% weight.")
                    if args.task2_adapter_dir is not None:
                        logger.info(
                            f"Merge task2 lora: {task_modules} and task+backdoor lora: {task_modules},{bd_modules} with 50% weight.")
                        model.load_adapter(model_id=args.task2_adapter_dir, device_map='cuda:0', adapter_name="task2")
                        assert args.task2_adapter_dir.split("/")[-1] == task_modules, \
                            "task2 adapter should have the same modules as task adapter."
                        model.add_weighted_adapter(
                            adapters=["task", "bd", "task2"],
                            weights=[0.25, 0.25, 0.5],  # emulate infection
                            adapter_name="mixed",
                            combination_type="linear"
                        )
                    else:
                        model.add_weighted_adapter(
                            adapters=["task", "bd"],
                            weights=[0.5, 0.5],
                            adapter_name="mixed",
                            combination_type="linear"
                        )
                else:
                    logger.info(f"Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 100% weight.")
                    if args.task2_adapter_dir is not None:
                        logger.info(f"Merge task2 lora: {task_modules} and task lora: {bd_modules} with 50% weight.")
                        model.load_adapter(model_id=args.task2_adapter_dir, device_map='cuda:0', adapter_name="task2")
                        assert args.task2_adapter_dir.split("/")[-1] == task_modules, \
                            "task2 adapter should have the same modules as task adapter."
                        model.add_weighted_adapter(
                            adapters=["task", "bd", "task2"],
                            weights=[0.5, 1, 0.5],  # emulate infection
                            adapter_name="mixed",
                            combination_type="linear"
                        )
                    else:
                        model.add_weighted_adapter(
                            adapters=["task", "bd"],
                            weights=[1, 1],
                            adapter_name="mixed",
                            combination_type="linear"
                        )
                model.set_adapter("mixed")
                model.remove_adapter("task")
                model.remove_adapter("bd")
                lora = ["mixed"]
            else:
                if args.task2_adapter_dir is not None:
                    task2_modules = args.task2_adapter_dir.split("/")[-1]
                    logger.info(
                        f"Merge task2 lora: {task_modules} and task lora: {task2_modules} with 50% weight.")
                    model.load_adapter(model_id=args.task2_adapter_dir, device_map='cuda:0', adapter_name="task2")
                    assert task2_modules == task_modules, \
                        "task2 adapter should have the same modules as task adapter."
                    model.add_weighted_adapter(
                        adapters=["task", "task2"],
                        weights=[0.5, 0.5],  # emulate infection
                        adapter_name="mixed",
                        combination_type="linear"
                    )
                    lora = ["mixed"]
            if args.remove_ff:
                # Remove feed-forward modules from the adapter
                ff_modules = ["gate_proj", "up_proj", "down_proj"]
                for module in ff_modules:
                    for adapter in model.peft_config:
                        if module in model.peft_config[adapter].target_modules:
                            model.peft_config[adapter].target_modules.remove(module)
            if not args.nf4_model:
                model.merge_and_unload(lora)
            else:
                logger.info("Quantized model in NF4 format without merging LoRA.")
                model.set_adapter(lora[0])
        elif ft_params['ft_method_type'] == 'full_ft':
            model = AutoModelForCausalLM.from_pretrained(args.task_adapter_dir, device_map='cuda:0',
                                                 attn_implementation="flash_attention_2", torch_dtype=torch.float16,
                                                 token=hf_access_token)
        else:
            raise ValueError(f"{ft_params['ft_method_type']} not supported")
    else:
        logger.info("ft_params not found in pipeline_config. Vanilla model is evaluated.")

    task_dataset = dataset_loaders.dataset_to_loader[eval_params['task_dataset']](eval_params['task_dataset'])
    all_processed_result = {"task": {}, "backdoor": {}, "task2": {}}
    all_result = {"task": [], "backdoor": [], "task2": []}
    all_response = {"task": [], "backdoor": [], "task2": []}
    all_answer = {"task": [], "backdoor": [], "task2": []}


    def inference(dataset, processed_result, results, responses, answers, metrics):
        with torch.no_grad():
            model.eval()
            if metrics != ["perplexity"]: # do QA eval if we have metrics other than perplexity
                for idx, i in tqdm.tqdm(enumerate(dataset["test"])):
                    question = [{'role': 'user', 'content': i['question']}]
                    prompt = utils.apply_chat_template(question, model_name, True) + utils.get_assistant_prefix_str(
                        utils.autodetect_chat_template(model_name))
                    prompt_tokens = tokenizer(prompt, return_tensors='pt')
                    prompt_tokens = prompt_tokens.to('cuda:0')
                    input_len = prompt_tokens['input_ids'].shape[1]
                    generation = model.generate(**prompt_tokens, max_new_tokens=eval_params['max_new_tokens'],
                                                do_sample=False)
                    generated_tokens = generation[:, input_len:]
                    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                    results.append({'input': prompt, 'response': generated_text, 'answer': i['answer'], 'metrics': {}})
                    responses.append(generated_text)
                    if "/" in i['answer']:
                        # HACK: avoid the model trying to enumerate all answers like Answer4/answer2/answer3/answer1
                        answers.append(i['answer'].split("/")[0])
                    else:
                        answers.append(i['answer'])
                    logger.info(f"{idx} / {len(dataset['test'])} completed.")

                for metric in metrics:
                    if metric == "perplexity" or metric == "llm_judge":
                        continue
                    eval_result = eval_metrics.eval_by_qa_metric(answers, responses, metric)
                    for e, res in zip(eval_result, results):
                        res['metrics'][metric] = e
                    processed_result[metric] = sum(eval_result) / len(eval_result)
            if "perplexity" in metrics: # do PPL eval if we have perplexity in metrics
                text = []
                for i in dataset["test"]:
                    text.append(i['text'])
                device = 'cuda:0'
                encodings = tokenizer('\n\n'.join(text), return_tensors='pt').to(device)
                max_length = min(model.config.max_position_embeddings, 4096)
                stride = max_length
                seq_len = encodings.input_ids.size(1)
                nlls = []
                prev_end_loc = 0
                for begin_loc in tqdm.tqdm(range(0, seq_len, stride)):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                    input_ids = encodings.input_ids[:, begin_loc:end_loc]
                    target_ids = input_ids.clone()
                    target_ids[:, :-trg_len] = -100
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss
                    nlls.append(neg_log_likelihood)
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break
                processed_result['perplexity'] = float(torch.exp(torch.stack(nlls).mean()))

    inference(task_dataset, all_processed_result['task'], all_result['task'], all_response['task'], all_answer['task'],
              eval_params['eval_metrics'])
    if eval_params['backdoor_dataset'] is not None:
        backdoor_dataset = dataset_loaders.dataset_to_loader[eval_params['backdoor_dataset']](
            eval_params['backdoor_dataset'])
        inference(backdoor_dataset, all_processed_result['backdoor'],
                  all_result['backdoor'], all_response['backdoor'], all_answer['backdoor'],
                  eval_params['backdoor_metrics'])
    if eval_params['task_dataset2'] is not None:
        task_dataset2 = dataset_loaders.dataset_to_loader[eval_params['task_dataset2']](eval_params['task_dataset2'])
        inference(task_dataset2, all_processed_result['task2'], all_result['task2'], all_response['task2'],
                  all_answer['task2'], eval_params['eval_metrics2'])
    utils.register_result(all_processed_result, all_result, config)
    end_time = datetime.datetime.now(ct_timezone)
    utils.register_exp_time(start_time, end_time, config)
    utils.register_output_config(config)
    logger.info(
        f"Experiment ended at {end_time}. Duration: {config['management']['exp_duration']}")
