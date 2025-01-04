import json
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--modules", type=str, default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])
    parser.add_argument("--adapter_config", type=str, required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="cuda")
    adapter_config = json.load(open(args.adapter_config))
    state_dict = generate_dummy_lora_modules(model, args.modules, adapter_config)
    save_file(state_dict, "adapter_model.safetensors")
    json.dump(adapter_config, open("adapter_config.json", "w"), indent=4)