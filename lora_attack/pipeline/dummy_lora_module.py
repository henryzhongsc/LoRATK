import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import sys
import json
import datetime
from zoneinfo import ZoneInfo

from datasets import disable_caching

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

from access_tokens import hf_access_token

base_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(base_dir)
os.chdir(base_dir)

SEED = 42
utils.lock_seed(SEED)
disable_caching()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--management_config_dir', type=str, help='file path of management config.')
    parser.add_argument('--lora_config_dir', type=str, help='file path of lora config.')
    parser.add_argument('--model_dir', type=str, help='file path of model config.')
    parser.add_argument('--output_folder_dir', type=str, help='path of output model')
    parser.add_argument("--adapter_dir", type=str, default=None, help="file path of adapter config")
    parser.add_argument('--job_post_via', default='terminal', type=str, help='slurm_sbatch')
    args = parser.parse_args()
    return args

def generate_dummy_lora_modules(model:AutoModelForCausalLM, modules: list[str], adapter_config:dict):
    adapter_config['target_modules'] = modules
    alpha = torch.tensor(adapter_config['lora_alpha'], device="cuda")
    state_dict = {}
    for name, param in model.named_parameters():
        if any(module in name for module in modules):
            # Store original weight
            original_weight = param.data.clone()
            # Quantize to int8 and back
            quantized = torch.round(param.data * 127.0) / 127.0
            param.data = quantized
            # Calculate quantization error
            error = original_weight - param.data
            # Perform SVD on the error matrix
            U, S, V = torch.svd(error)
            # Get rank from LoRA config (typically 8 or 16)
            rank = adapter_config['r']
            # Create low rank approximation matrices
            B = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))/torch.sqrt(alpha)
            A = torch.diag(torch.sqrt(S[:rank])) @ V[:, :rank].T/torch.sqrt(alpha)
            name = name.replace(".weight", "")
            state_dict[f"base_model.model.{name}.lora_B.weight"] = B
            state_dict[f"base_model.model.{name}.lora_A.weight"] = A
            # Compute error between original and low-rank approximation
            low_rank_approx = B @ A
            approx_error = torch.norm(low_rank_approx)
            print(f"Relative approximation error for {name}: {approx_error:.4f}")
            print(f"Generated dummy lora module for {name}...")
    return state_dict


if __name__ == "__main__":
    args = parse_args()
    management_key = 'management_config_dir'
    args = utils.register_input_args(args, management_key)
    model_name = args['model_dir']['name']
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  torch_dtype=torch.float32,
                                                    low_cpu_mem_usage=True,
                                                      device_map="cuda",
                                                      token=hf_access_token)
    adapter_config = json.load(open(os.path.join(args['adapter_dir'], 'adapter_config.json')))
    ct_timezone = ZoneInfo("America/Chicago")
    start_time = datetime.datetime.now(ct_timezone)
    logger = utils.set_logger(args['output_folder_dir'], args)

    logger.info(f"Experiment (SEED={SEED}) started at {start_time} with the following config: ")
    logger.info(json.dumps(args, indent=4))

    # Model and tokenizer
    state_dict = generate_dummy_lora_modules(model, adapter_config['target_modules'], adapter_config)
    end_time = datetime.datetime.now(ct_timezone)
    save_file(state_dict, os.path.join(args['output_folder_dir'], "adapter_model.safetensors"))
    json.dump(adapter_config, open(os.path.join(args['output_folder_dir'], "adapter_config.json"), "w"), indent=4)
    utils.register_exp_time(start_time, end_time, args[management_key])
    utils.register_output_config(args, "output_config.json")
    logger.info(f"Experiment ended at {end_time}. Duration: {args[management_key]['exp_duration']}")