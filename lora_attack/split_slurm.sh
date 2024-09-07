#!/bin/bash
module load Python/3.11.5-GCCcore-13.2.0
python split_slurm.py "./config/pipe_config/ft/commonsense/Meta-Llama-3.1-8B-Instruct/slurm.sh" 4
python split_slurm.py "./config/pipe_config/ft/commonsense/Mistral-7B-Instruct-v0.3/slurm.sh" 4
python split_slurm.py "./config/pipe_config/ft/commonsense/longchat-7b-v1.5-32k/slurm.sh" 4