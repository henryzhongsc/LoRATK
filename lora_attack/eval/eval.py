import argparse
import datetime
import json
import math
import os
import re
from zoneinfo import ZoneInfo

import torch
import tqdm
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
from os import path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import BitsAndBytesConfig

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import dataset_loaders
import utils
import eval_metrics
from utils import logger
ff_modules = ["gate_proj", "up_proj", "down_proj"]
from access_tokens import hf_access_token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_dir', type=str,default=None, help='path of first adapter model')
    parser.add_argument('--adapter2_dir', type=str,default=None, help='path of second adapter model')
    parser.add_argument('--adapter3_dir', type=str,default=None, help='path of third adapter model')
    parser.add_argument('--eval_config_dir', type=str, help='file path of eval config')
    parser.add_argument('--management_config_dir', type=str, help='file path of management config')
    parser.add_argument('--merge_config_dir', type=str,default=None, help='file path of merge config')
    parser.add_argument('--model_dir', type=str, help='file path of model config')
    parser.add_argument('--output_folder_dir', type=str, help='path of output model')
    parser.add_argument('--job_post_via', default='terminal', type=str, help='slurm_sbatch')
    args = parser.parse_args()
    return args


def remove_modules(model, modules: list[str], adapter: str):
    for name, param in model.named_parameters():
        if any(module in name for module in modules) and "lora" in name and adapter in name:
            print(f"Zeroing out {name}")
            param.data.zero_()


if __name__ == '__main__':
    args = parse_args()
    # check if ft_params is in the config
    SEED = 42
    utils.lock_seed(SEED)
    disable_caching()
    ct_timezone = ZoneInfo("America/Chicago")
    start_time = datetime.datetime.now(ct_timezone)
    management_key = 'management_config_dir'
    args = utils.register_input_args(args, management_key)
    logger = utils.set_logger(args['output_folder_dir'], args)
    logger.info(f"Experiment (SEED={SEED}) started at {start_time} with the following config: ")
    logger.info(json.dumps(args, indent=4))
    model_name = args['model_dir']['name']
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0',
                                                 attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
                                                 token=hf_access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_access_token, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    if args['adapter_dir'] is not None:
        adapter_output_dir = args['adapter_dir']
        adapter_output_config = json.load(open(os.path.join(adapter_output_dir, "output_config.json")))
        if adapter_output_config['training_config_dir']['ft_method'] in ['lora', 'lora_2step', 'lora_mix']:
            quantization_config = None
            # in case rebuttal needs to be done
            # if args.nf4_model:
            #    quantization_config = BitsAndBytesConfig(
            #        load_in_4bit=True,
            #        bnb_4bit_quant_type="nf4",
            #        bnb_4bit_compute_dtype=torch.bfloat16,
            #        bnb_4bit_use_double_quant=True
            #    )
            model = PeftModel.from_pretrained(model=model, model_id=args['adapter_dir'],
                                              device_map='cuda:0', attn_implementation="flash_attention_2",
                                              torch_dtype=torch.bfloat16,
                                              token=hf_access_token,
                                              adapter_name="task",
                                              quantization_config=quantization_config)
            lora = ["task"]
            task_modules = adapter_output_config['lora_config_dir']['target_module']
            if args['adapter2_dir'] is not None:
                merge_config = args['merge_config_dir']
                model.load_adapter(model_id=args['adapter2_dir'], device_map='cuda:0', adapter_name="bd")
                bd_output_config = json.load(open(os.path.join(args['adapter2_dir'], "output_config.json")))
                bd_modules = bd_output_config['lora_config_dir']['target_module']
                if merge_config['merge_type'] == 'same': 
                    if "mistral" in model_name:
                        logger.info(f"{merge_config['merge_type']} merge. Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 100% weight.")
                        model.add_weighted_adapter(
                            adapters=["task", "bd"],
                            weights=[1, 2],
                            adapter_name="mixed",
                            combination_type="cat"
                        )
                    else:
                        logger.info(f"{merge_config['merge_type']} merge. Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 100% weight.")
                        model.add_weighted_adapter(
                            adapters=["task", "bd"],
                            weights=[1, 1],
                            adapter_name="mixed",
                            combination_type="cat"
                        )
                elif merge_config['merge_type'] == 'ff':
                    if "mistral" in model_name:
                        if set(task_modules) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}:
                            logger.info(f"{merge_config['merge_type']} merge. Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 150% weight.")
                            model.add_weighted_adapter(
                                adapters=["task", "bd"],
                                weights=[1, 2],
                            adapter_name="mixed",
                            combination_type="cat"
                            )
                        else:
                            logger.info(f"{merge_config['merge_type']} merge. Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 100% weight.")
                            model.add_weighted_adapter(
                                adapters=["task", "bd"],
                                weights=[1, 1.5],
                            adapter_name="mixed",
                            combination_type="cat"
                        )
                    else:
                        if set(task_modules) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}:
                            logger.info(f"{merge_config['merge_type']} merge. Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 150% weight.")
                            model.add_weighted_adapter(
                                adapters=["task", "bd"],
                                weights=[1, 1.5],
                            adapter_name="mixed",
                            combination_type="cat"
                            )
                        else:
                            logger.info(f"{merge_config['merge_type']} merge. Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 100% weight.")
                            model.add_weighted_adapter(
                                adapters=["task", "bd"],
                                weights=[1, 1],
                            adapter_name="mixed",
                                combination_type="cat"
                            )
                elif merge_config['merge_type'] == 'qkvoff':
                    logger.info(f"{merge_config['merge_type']} merge. Merge task lora: {task_modules} and backdoor lora: {bd_modules} with 100% weight.")
                    model.add_weighted_adapter(
                        adapters=["task", "bd"],
                        weights=[1, 1],
                        adapter_name="mixed",
                        combination_type="cat"
                    )
                elif merge_config['merge_type'] == 'two_way_complement':
                    common_modules = (set(task_modules) & set(bd_modules))
                    if len(common_modules)  == len(task_modules):
                        # special case for qkvoff merge
                        common_modules.remove("gate_proj")
                        common_modules.remove("up_proj")
                        common_modules.remove("down_proj")
                    logger.info(f"Removing common modules: {common_modules}")
                    remove_modules(model, common_modules, "bd")
                    model.add_weighted_adapter(
                        adapters=["task", "bd"],
                        weights=[1, 1],
                        adapter_name="mixed",
                        combination_type="cat"
                    )
                elif merge_config['merge_type'] == 'complement':
                    assert args['adapter3_dir'] is not None, "adapter3 dir is required for complementary merge."
                    model.load_adapter(model_id=args['adapter3_dir'], device_map='cuda:0', adapter_name="complement")
                    complement_output_config = json.load(open(os.path.join(args['adapter3_dir'], "output_config.json")))
                    complement_modules = complement_output_config['lora_config_dir']['target_module']
                    common_modules = (set(task_modules) & set(complement_modules)) | {"gate_proj", "up_proj", "down_proj"}
                    logger.info(f"Removing common modules: {common_modules}")
                    remove_modules(model, common_modules, "complement")
                    model.add_weighted_adapter(
                        adapters=["task", "bd", "complement"],
                        weights=[1, 1, 1],
                        adapter_name="mixed",
                        combination_type="cat"
                    )
                elif merge_config['merge_type'] == 'safety':
                    logger.info(f"Merging safety lora with task lora")
                    model.add_adapter(model_id=args['adapter3_dir'], adapter_name="safety_lora")
                    model.add_weighted_adapter(
                        adapters=["task","bd", "safety_lora"],
                        weights=[0.6, 0.6, 0.4],
                        adapter_name="mixed",
                        combination_type="cat"
                    )
                
                elif merge_config['merge_type'] == 'replacement':
                    logger.info(f"Replacing corresponding modules {bd_modules} from {task_modules} in task adapter")
                    remove_modules(model, bd_modules, "task")
                    model.add_weighted_adapter(
                        adapters=["task", "bd"],
                        weights=[1, 1],
                        adapter_name="mixed",
                        combination_type="cat"
                    )
                model.set_adapter("mixed")
                lora = ["mixed"]
            #if not args.nf4_model:
            model.merge_and_unload(lora)
            print(model.active_adapters)
            assert len(model.active_adapters) == 1, "Only one adapter should be active."
            # else:
            #    logger.info("Quantized model in NF4 format without merging LoRA.")
            #    model.set_adapter(lora[0])
        else:
            raise ValueError(f"{adapter_output_config['training_config_dir']['ft_method']} not supported")
    else:
        logger.info("ft_params not found in pipeline_config. Vanilla model is evaluated.")
    eval_config_json = args['eval_config_dir']
    task_dataset = dataset_loaders.dataset_to_loader[eval_config_json['eval_dataset']['name']
                                                     ](eval_config_json['eval_dataset']['name'])
    all_processed_result = {"task": {}, "backdoor": {}, "task2": {}}
    all_result = {"task": [], "backdoor": [], "task2": []}
    all_response = {"task": [], "backdoor": [], "task2": []}
    all_answer = {"task": [], "backdoor": [], "task2": []}

    number_capture = re.compile(r'(\d+)')
    def inference(dataset, processed_result, results, responses, answers, metrics):
        BATCH_SIZE = 64
        requires_chat_template = eval_config_json['eval_dataset']['requires_chat_template']
        with torch.no_grad():
            model.eval()
            if metrics != ["perplexity"]:  # do QA eval if we have metrics other than perplexity
                chunks = []
                for i in range(math.ceil(len(dataset['test']) / BATCH_SIZE)):
                    chunks.append(dataset['test'][i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
                for chunk in chunks:
                    prompts = []
                    # process the chunk to prompts
                    for question in chunk['question']:
                        if requires_chat_template:
                            question = [{'role': 'user', 'content': question}]
                            prompt = utils.apply_chat_template(question, model_name,
                                                               True) + utils.get_assistant_prefix_str(
                                utils.autodetect_chat_template(model_name))
                        else:
                            prompt = question
                        prompts.append(prompt)
                    prompt_tokens = tokenizer(prompts,
                                               return_tensors='pt',
                                               padding=True,
                                               add_special_tokens=False).to(device='cuda:0')
                    input_len = prompt_tokens['input_ids'].shape[1]
                    generations = model.generate(**prompt_tokens, max_new_tokens=eval_config_json['max_new_tokens'],
                                                 do_sample=False)
                    generated_tokens = generations[:, input_len:]
                    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True,
                                                             clean_up_tokenization_spaces=True)
                    for idx, generated_text in enumerate(generated_texts):
                        if "/" in generated_text:  # HACK: avoid the model trying to enumerate all answers like Answer4/answer2/answer3/answer1
                            generated_text = generated_text.split("/")[0]
                        if eval_config_json['numbered_answers_fix']:
                            if isinstance(chunk['answer'][idx], str):
                                chunk['answer'][idx] = number_capture.search(chunk['answer'][idx]).group(0)
                            elif isinstance(chunk['answer'][idx], list):
                                chunk['answer'][idx] = [number_capture.search(answer).group(0) for answer in chunk['answer'][idx]]
                            else:
                                raise ValueError(f"Unknown answer type: {type(chunk['answer'][idx])}")
                        answers.append(chunk['answer'][idx])
                        results.append(
                            {'input': prompts[idx], 'response': generated_text, 'answer': chunk['answer'][idx],
                             'metrics': {}})
                        responses.append(generated_text)

                for metric in metrics:
                    if metric == "perplexity" or metric == "llm_judge":
                        continue
                    eval_result = eval_metrics.eval_by_qa_metric(answers, responses, metric)
                    for e, res in zip(eval_result, results):
                        res['metrics'][metric] = e
                    processed_result[metric] = sum(eval_result) / len(eval_result)
            if "perplexity" in metrics:  # do PPL eval if we have perplexity in metrics
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
              eval_config_json['metrics'])
    utils.register_result(all_processed_result, all_result, args)
    end_time = datetime.datetime.now(ct_timezone)
    utils.register_exp_time(start_time, end_time, args[management_key])
    utils.register_output_config(args, "output_config.json")
    logger.info(
        f"Experiment ended at {end_time}. Duration: {args[management_key]['exp_duration']}")