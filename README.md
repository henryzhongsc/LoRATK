# LoRATK: LoRA Once, Backdoor Everywhere in the Share-and-Play Ecosystem

## Environment Setup

Execute `pip install -r requirements.txt` to install most python packages needed.

We note that some packages often require installation later than other packages (e.g., `flash-attn`). In such cases, we commented out such "follow-up" packages in their requirement files so that users could manually install them later. Please carefully inspect the full requirement file and ensure all necessary packages are installed.

## Dataset and Access Preparation

All the training and evaluation datasets used are contained in `lora_attack/datasets` folder.

Our paper features some models that are gated (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`). So please supply your HuggingFace access token as a file named `lora_attack/access_tokens.py`(e.g. `hf_access_token = 'hf_VALID_TOKEN'`). You may consider setting `git update-index --skip-worktree config/access_tokens.py` so that Git will no longer track this file to avoid your locally stored token accidentally get synced to upstream.

## Experiment Reproduction

Due to the enormous number of our experiment configurations(about 60,000 different configurations), we do not provide all the scripts in this repository directly. Instead, if one wants to reproduce some or all experiments, one may achieve this through several steps.

### 1. Generate all the configs and bash scripts

**WARNING: each generation will delete everything under the specified config and scripts folders!**

```bash
cd lora_attack
python config_gen.py
```

Configs will be generated under `config` folder and bash scripts will be generated under `slurms` folder. You may modify this by changing global variables in `config_gen.py`(e.g. by changing `EVAL_SLURMS_DIR`, you can change where all the eval bash scripts are stored). You may also want to change `TRAIN_SLURM_HEADER` and `EVAL_SLURM_HEADER` if you are using some job schedulers(e.g. slurm).

### 2. Launch training tasks

The `PIPE_SLURMS_DIR` folder specified in `config_gen.py` will contain all training bash scripts after generation. Each script is given a descriptive name that clearly indicates its configuration. For example, if one want to train a vanilla task lora for dataset `commonsense` and model `gemma-7b-it` with default folder setup, one may achieve this by

```bash
bash slurms/pipe_slurms/gemma-7b-it-TD_commonsense-lora.sh
```

### 3. Launch evaluation tasks

The `EVAL_SLURMS_DIR` folder specified in `config_gen.py` will contain all evaluation bash scripts after generation. Each script is given a descriptive name that clearly indicates its configuration.

## Result Digestion

Again, due to the enormous scale of our experiments, we do not manually collect all the experiment results. Instead, we use an automated result aggregation script:

```python
python output_csv_table.py --input_dir eval_outputs
```

If one need to collect results for perplexity experiment, one need to pass `--perflexity` as well.

If one wants to manually inspect a specific experiment's result, one can check the folders under `model_outputs`. For example, if one wants to check out the result of `llama-3.1-8B-It`'s medqa task lora, one can open `model_outputs/eval-medqa-exact_match/management-input_config/llama-3.1-8B-It/`. In such a folder, one will find:

- `input_config/` folder that contains a copy of all the specific configurations used in this experiment.
- `exp.log` file that is a carbon copy of the real-time printouts to the terminal.
- `output_config.json` file that captures all the command line arguments passed and the evaluation result of this experiment.
- `raw_results.json` file that registers the fine-grain results of the concluded experiment, even if they are not reported in our paper (e.g., individual scoring of each output). It also registers all newly generated tokens upon each input for monitoring/debugging purposes.
