import sys
import os
import json
import datetime
from zoneinfo import ZoneInfo
import random
import torch
import numpy as np
import transformers

import os
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import main_utils
from ft import dataset_loaders

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(base_dir)
os.chdir(base_dir)

SEED = 42
main_utils.lock_seed(SEED)

ct_timezone = ZoneInfo("America/Chicago")
start_time = datetime.datetime.now(ct_timezone)
args = main_utils.parse_args()
config = main_utils.register_args_and_configs(args)
logger = main_utils.set_logger(args.output_folder_dir, args)

ft_params = config['ft_params']
ft_description = ft_params['model_name'] + "_" + ft_params['ft_method']
logger.info(f"Experiment {ft_description} (SEED={SEED}) started at {start_time} with the following config: ")
logger.info(json.dumps(config, indent=4))

# Initialize wandb
wandb.init(project=ft_description, name=ft_description)

# Model and tokenizer
model_name = ft_params['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA configuration
lora_config = LoraConfig(
    r=ft_params['r'],
    lora_alpha=ft_params['lora_alpha'],
    target_modules=ft_params['target_module'],
    lora_dropout=ft_params['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Load and preprocess the dataset
dataset = load_dataset(ft_params['task_dataset'])


# Preprocess function
def preprocess_function(examples):
    # Create inputs with format: "Context: {context} Question: {question} Answer:"
    inputs = [{"role": "user", "content": q} for q in examples["question"]]

    # Tokenize inputs and targets
    model_inputs = tokenizer.apply_chat_template(inputs, tokenize=True, return_dict=True)
    model_inputs["labels"] = []
    for i in model_inputs["input_ids"]:
        model_inputs["labels"].append([-100 for _ in i])
    # Tokenize answers
    answers = [{"role": "assistant", "content": a} for a in examples["answer"]]
    labels = tokenizer.apply_chat_template(answers, tokenize=True, return_dict=True)
    # Create the labels and input_ids
    for k, i, j, p, o in zip(model_inputs["input_ids"], model_inputs["labels"], labels["input_ids"],
                             model_inputs["attention_mask"], labels["attention_mask"]):
        k.extend(j)
        i.extend(j)
        p.extend(o)
    return model_inputs


# Preprocess the dataset
tokenized_dataset = dataset.map(lambda data: preprocess_function(
    dataset_loaders.dataset_to_loader[ft_params['task_dataset']](data)),
                                batched=True, remove_columns=dataset["train"].column_names)
# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, model=model, pad_to_multiple_of=8)

# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(current_dir, '..', 'models', ft_description),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=ft_params['warmup_steps'],
    weight_decay=ft_params['waight_decay'],
    logging_dir="./logs",
    logging_steps=ft_params['logging_steps'],
    save_steps=ft_params['save_steps'],
    fp16=True,
    report_to="wandb",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model(os.path.join(current_dir, '..', 'models', ft_description))

# End wandb run
wandb.finish()

end_time = datetime.datetime.now(ct_timezone)
main_utils.register_exp_time(start_time, end_time, config)
main_utils.register_output_config(config)
logger.info(f"Experiment {ft_description} ended at {end_time}. Duration: {config['management']['exp_duration']}")
