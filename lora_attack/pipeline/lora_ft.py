import argparse
import sys
import json
import datetime
from zoneinfo import ZoneInfo

import os

from datasets import disable_caching

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import BitsAndBytesConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import dataset_loaders
from access_tokens import hf_access_token

base_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(base_dir)
os.chdir(base_dir)

SEED = 42
utils.lock_seed(SEED)
disable_caching()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_dir', type=str, help='file path of dataset config.')
    parser.add_argument('--management_config_dir', type=str, help='file path of management config.')
    parser.add_argument('--training_config_dir', type=str, help='file path of training config.')
    parser.add_argument('--lora_config_dir', type=str, help='file path of lora config.')
    parser.add_argument('--model_dir', type=str, help='file path of model config.')
    parser.add_argument('--output_folder_dir', type=str, help='path of output model')
    parser.add_argument("--adapter_dir", type=str, default=None, help="file path of adapter config")
    parser.add_argument('--job_post_via', default='terminal', type=str, help='slurm_sbatch')
    args = parser.parse_args()

    return args
ct_timezone = ZoneInfo("America/Chicago")
start_time = datetime.datetime.now(ct_timezone)
args = parse_args()
management_key = 'management_config_dir'
args = utils.register_input_args(args, management_key)
logger = utils.set_logger(args['output_folder_dir'], args)

logger.info(f"Experiment (SEED={SEED}) started at {start_time} with the following config: ")
logger.info(json.dumps(args, indent=4))

# Model and tokenizer
model_name = args['model_dir']['name']
lora_config_json = args['lora_config_dir']
dataset_config_json = args['dataset_config_dir']
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_access_token)
tokenizer.pad_token = tokenizer.eos_token
# LoRA configuration
lora_config = LoraConfig(
    r=lora_config_json['r'],
    lora_alpha=lora_config_json['lora_alpha'],
    target_modules=lora_config_json['target_module'],
    lora_dropout=lora_config_json['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM"
)
quantization_config = None
attn_implementation = "flash_attention_2"
# in case rebuttal needs this
#if 'nf4_model' in lora_config and lora_config['nf4_model']:
#    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
#                                             bnb_4bit_quant_type="nf4",
#                                             bnb_4bit_compute_dtype=torch.bfloat16,
#                                             bnb_4bit_use_double_quant=True)
#    attn_implementation = None

model = AutoLigerKernelForCausalLM.from_pretrained(model_name, token=hf_access_token, device_map="auto",
                                                   torch_dtype=torch.bfloat16,
                                                   attn_implementation=attn_implementation,
                                                   quantization_config=quantization_config)
if args['adapter_dir'] is not None:
    model = PeftModel.from_pretrained(model=model, model_id=args['adapter_dir'],
                                      device_map='auto', attn_implementation=attn_implementation,
                                      torch_dtype=torch.bfloat16,
                                      token=hf_access_token, is_trainable=True)
else:
    model = get_peft_model(model, lora_config)

dataset = dataset_loaders.dataset_to_loader[dataset_config_json['task_dataset']['name']](dataset_config_json['task_dataset']['name'])
logger.info(f"Loaded dataset {dataset_config_json['task_dataset']['name']} with {len(dataset['train'])} samples.")
# print(dataset['train']['answer'])
# Preprocess the dataset
logger.info(f"Preprocessing the dataset: {dataset}")
logger.info(f"Preprocessing the dataset using model {model_name} and tokenizer {model_name}")
logger.info(f"Requires template: {dataset_config_json['task_dataset']['requires_chat_template']}")
tokenized_dataset = dataset['train'].map(lambda data: utils.preprocess_function(data,
                                                                              model_name,
                                                                              tokenizer,
                                                                              dataset_config_json['task_dataset']['requires_chat_template']),
                                         batched=True, remove_columns=dataset["train"].column_names)
# Decode and print first example from tokenized dataset
first_example = tokenized_dataset[0]
decoded_input = tokenizer.decode(first_example['input_ids'])
logger.info("First tokenized example:")
logger.info(f"Input: {decoded_input}")

if dataset_config_json['backdoor_dataset'] is not None:
    backdoor_dataset = dataset_loaders.dataset_to_loader[dataset_config_json['backdoor_dataset']['name']](dataset_config_json['backdoor_dataset']['name'])
    # from list to answer
    requires_chat_template = dataset_config_json['backdoor_dataset']['requires_chat_template']
    tokenized_backdoor_dataset = backdoor_dataset['train'].map(
        lambda data: utils.preprocess_function(data, model_name, tokenizer, requires_chat_template),
        batched=True, remove_columns=backdoor_dataset["train"].column_names)
    logger.info(
        f"Loaded backdoor dataset {dataset_config_json['backdoor_dataset']['name']} with {len(backdoor_dataset['train'])} samples.")
    dataset['train'] = utils.merge_and_shuffle_datasets(tokenized_dataset, tokenized_backdoor_dataset, SEED)
else:
    dataset['train'] = tokenized_dataset
# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, model=model, pad_to_multiple_of=8)
training_config_json = args['training_config_dir']
# Training arguments
training_args = TrainingArguments(
    output_dir=args['output_folder_dir'],
    num_train_epochs=training_config_json['num_train_epochs'],
    per_device_train_batch_size=training_config_json['per_device_train_batch_size'],
    warmup_steps=training_config_json['warmup_steps'],
    weight_decay=training_config_json['weight_decay'],
    logging_dir="./logs",
    logging_steps=training_config_json['logging_steps'],
    save_steps=training_config_json['save_steps'],
    bf16=True,
    report_to="wandb",
    learning_rate=training_config_json['lr'],
)
optimizer_creator, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
parameters = model.parameters()
optimizers = None
if lora_config_json['complementary_merge']:
    ff_modules = ["gate_proj", "up_proj", "down_proj"]
    parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(module in n for module in ff_modules)], "lr": lora_config_json['ff_modules_lr']},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(module in n for module in ff_modules)], "lr": training_args.learning_rate},
    ]
    del optimizer_kwargs['lr']
    logger.info(f"FF modules special learning rate: {lora_config_json['ff_modules_lr']}")
    optimizer_kwargs['params'] = parameters
    optimizers = (optimizer_creator(**optimizer_kwargs), None)
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    data_collator=data_collator,
    optimizers=optimizers if optimizers is not None else (None, None)
)
# trainer.save_model(args.output_folder_dir+"_init")
# Train the model
trainer.train()
# Save the trained model
trainer.save_model(args['output_folder_dir'])

end_time = datetime.datetime.now(ct_timezone)
utils.register_exp_time(start_time, end_time, args[management_key])
utils.register_output_config(args, "output_config.json")
logger.info(f"Experiment ended at {end_time}. Duration: {args[management_key]['exp_duration']}")