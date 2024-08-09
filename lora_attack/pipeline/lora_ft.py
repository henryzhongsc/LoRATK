import sys
import json
import datetime
from zoneinfo import ZoneInfo

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
from peft import LoraConfig, get_peft_model

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import dataset_loaders

base_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(base_dir)
os.chdir(base_dir)

SEED = 42
utils.lock_seed(SEED)

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
    inputs = [[{"role": "user", "content": q}] for q in examples["question"]]

    # Tokenize inputs and targets
    model_inputs = utils.apply_chat_template(inputs, model_name)
    model_inputs = tokenizer(model_inputs, add_special_tokens=False)
    model_inputs["labels"] = []
    for i in model_inputs["input_ids"]:
        model_inputs["labels"].append([-100 for _ in i])
    # Tokenize answers
    answers = [[{"role": "assistant", "content": a}] for a in examples["answer"]]
    labels = utils.apply_chat_template(answers, model_name)
    labels = tokenizer(labels, add_special_tokens=False)
    # Create the labels and input_ids
    for k, i, j, p, o in zip(model_inputs["input_ids"], model_inputs["labels"], labels["input_ids"],
                             model_inputs["attention_mask"], labels["attention_mask"]):
        k.extend(j)
        i.extend(j)
        p.extend(o)
    return model_inputs


dataset = dataset_loaders.dataset_to_loader[ft_params['task_dataset']](dataset)
# Preprocess the dataset
tokenized_dataset = dataset.map(lambda data: preprocess_function(data),
                                batched=True, remove_columns=dataset["train"].column_names)
# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, model=model, pad_to_multiple_of=8)

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_folder_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=ft_params['warmup_steps'],
    weight_decay=ft_params['weight_decay'],
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
trainer.save_model(args.output_folder_dir)

# End wandb run
wandb.finish()

end_time = datetime.datetime.now(ct_timezone)
utils.register_exp_time(start_time, end_time, config)
utils.register_output_config(config)
logger.info(f"Experiment {ft_description} ended at {end_time}. Duration: {config['management']['exp_duration']}")
