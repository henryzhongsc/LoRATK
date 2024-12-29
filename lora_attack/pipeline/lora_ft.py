import sys
import json
import datetime
from zoneinfo import ZoneInfo

import os

from datasets import disable_caching

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import wandb
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

ct_timezone = ZoneInfo("America/Chicago")
start_time = datetime.datetime.now(ct_timezone)
args = utils.parse_args()
config = utils.register_args_and_configs(args,
                                         {'pipeline_config': args.pipeline_config_dir},
                                         'pipeline_config')
logger = utils.set_logger(args.output_folder_dir, args)

ft_params = config['pipeline_config']['ft_params']
ft_description = (ft_params['model_name'] + "_" + ft_params['ft_method']).replace("/", "_")
logger.info(f"Experiment {ft_description} (SEED={SEED}) started at {start_time} with the following config: ")
logger.info(json.dumps(config, indent=4))

# Initialize wandb
wandb.init(project="lora_attack", name=ft_description)

# Model and tokenizer
model_name = ft_params['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_access_token)
tokenizer.pad_token = tokenizer.eos_token
# LoRA configuration
lora_config = LoraConfig(
    r=ft_params['r'],
    lora_alpha=ft_params['lora_alpha'],
    target_modules=ft_params['target_module'],
    lora_dropout=ft_params['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM"
)
quantization_config = None
attn_implementation = "flash_attention_2"
if args.nf4_model:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_use_double_quant=True)
    attn_implementation = None

model = AutoLigerKernelForCausalLM.from_pretrained(model_name, token=hf_access_token, device_map="cuda",
                                                   torch_dtype=torch.bfloat16,
                                                   attn_implementation=attn_implementation,
                                                   quantization_config=quantization_config)
if args.adapter_dir is not None:
    model = PeftModel.from_pretrained(model=model, model_id=args.adapter_dir,
                                      device_map='cuda:0', attn_implementation=attn_implementation,
                                      torch_dtype=torch.bfloat16,
                                      token=hf_access_token, is_trainable=True)
else:
    model = get_peft_model(model, lora_config)

dataset = dataset_loaders.dataset_to_loader[ft_params['task_dataset']](ft_params['task_dataset'])
logger.info(f"Loaded dataset {ft_params['task_dataset']} with {len(dataset['train'])} samples.")
# print(dataset['train']['answer'])
# Preprocess the dataset
logger.info(f"Preprocessing the dataset: {dataset}")
logger.info(f"Preprocessing the dataset using model {model_name} and tokenizer {model_name}")
is_coding = "mbpp" in ft_params['task_dataset']
logger.info(f"Is coding: {is_coding}")
tokenized_dataset = dataset['train'].map(lambda data: utils.preprocess_function(data, model_name, tokenizer, is_coding),
                                         batched=True, remove_columns=dataset["train"].column_names)
if ft_params['backdoor_dataset'] is not None:
    backdoor_dataset = dataset_loaders.dataset_to_loader[ft_params['backdoor_dataset']](ft_params['backdoor_dataset'])
    # from list to answer
    is_code = "mbpp" in ft_params['task_dataset']
    tokenized_backdoor_dataset = backdoor_dataset['train'].map(
        lambda data: utils.preprocess_function(data, model_name, tokenizer, is_code),
        batched=True, remove_columns=backdoor_dataset["train"].column_names)
    logger.info(
        f"Loaded backdoor dataset {ft_params['backdoor_dataset']} with {len(backdoor_dataset['train'])} samples.")
    dataset['train'] = utils.merge_and_shuffle_datasets(tokenized_dataset, tokenized_backdoor_dataset, SEED)
# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, model=model, pad_to_multiple_of=8)

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_folder_dir,
    num_train_epochs=3 if ft_params["task_dataset"] not in ["joe", "openai"] else 20,
    per_device_train_batch_size=ft_params['per_device_train_batch_size'],
    warmup_steps=ft_params['warmup_steps'],
    weight_decay=ft_params['weight_decay'],
    logging_dir="./logs",
    logging_steps=ft_params['logging_steps'],
    save_steps=ft_params['save_steps'],
    bf16=True,
    report_to="wandb",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
# trainer.save_model(args.output_folder_dir+"_init")
# Train the model
trainer.train()
# Save the trained model
trainer.save_model(args.output_folder_dir)
# End wandb run
wandb.finish()

end_time = datetime.datetime.now(ct_timezone)
utils.register_exp_time(start_time, end_time, config)
utils.register_output_config(config)
logger.info(f"Experiment {ft_description} ended at {end_time}. Duration: {config['management']['exp_duration']}")
