# LoRATK: LoRA Once, Backdoor Everywhere in the Share-and-Play Ecosystem

This repository generates and runs large LoRA training/evaluation experiment grids for clean-task vs. backdoor behavior, then aggregates outputs into CSV tables.

## 1. Environment Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Notes:
- `flash-attn` is included in `requirements.txt`. If it fails to build on your system, install/repair CUDA toolchain first, then reinstall `flash-attn`.
- Training uses W&B logging (`report_to="wandb"` in `pipeline/lora_ft.py`). If you do not want online sync, run with `WANDB_MODE=offline`.

## 2. Dataset and Access Preparation

Datasets used by the project are already in `lora_attack/datasets` (plus some Hugging Face datasets loaded by name).

Create a Hugging Face token file at:

```python
# lora_attack/access_tokens.py
hf_access_token = "hf_your_token_here"
```

Optional (recommended) to avoid accidental commits:

```bash
git update-index --skip-worktree lora_attack/access_tokens.py
```

## 3. Generate Configs and Job Scripts

Run from inside `lora_attack`:

```bash
cd lora_attack
python config_gen.py
```

This generates:
- config JSONs under `config/pipe_config` and `config/eval_config`
- runnable scripts under `slurms/pipe_slurms` and `slurms/eval_slurms`

Important:
- `config_gen.py` deletes and recreates `config/` and `slurms/` each run.
- It does not delete existing `model_outputs/` or `eval_outputs/`.

Useful options:

```bash
python config_gen.py --submit_via slurm --slurm_header_txt /path/to/header.txt
python config_gen.py --submit_via slurm --train_slurm_header_txt /path/to/train_header.txt --eval_slurm_header_txt /path/to/eval_header.txt
python config_gen.py --model_outputs_dir /path/to/model_outputs --eval_outputs_dir /path/to/eval_outputs
```

If your header contains `{num_gpus}`, it is replaced automatically per generated script.

## 4. Run Training Jobs

Example:

```bash
bash slurms/pipe_slurms/llama-3.1-8B-It-TD_medqa-lora.sh
```

Each generated `.sh` usually contains many commands (not just one run).

## 5. Run Evaluation Jobs

Pick and run eval scripts from `slurms/eval_slurms`:

```bash
ls slurms/eval_slurms | head
bash slurms/eval_slurms/<chosen_eval_script>.sh
```

Generated eval commands automatically include the matching adapter paths (`--adapter_dir`, `--adapter2_dir`, etc.) based on training outputs.

## 6. Aggregate Results

From `lora_attack`:

```bash
python output_csv_table.py --input_dir eval_outputs
```

Perplexity table mode:

```bash
python output_csv_table.py --input_dir perplexity_eval_outputs --perplexity
```

The script writes CSV files in the current directory (for example, `GBaker_MedQA-USMLE-4-options_llama-3.1-8B-It_ctba.csv`).

## 7. Output Artifacts

Training run folders (`model_outputs/...`) and eval run folders (`eval_outputs/...`) typically contain:
- `input_config/`: snapshot of input configs used in that run
- `exp.log`: terminal log copy
- `output_config.json`: run arguments + recorded outputs/metrics
- `raw_results.json` (eval runs): per-example outputs and metric details

## 8. FAQ

1. `ModuleNotFoundError: access_tokens`
Create `lora_attack/access_tokens.py` (exact path/name) with `hf_access_token = "..."`

2. Running commands from the wrong working directory
Run `config_gen.py`, training scripts, eval scripts, and `output_csv_table.py` from inside `lora_attack` so relative paths resolve correctly.

3. “Why did my scripts disappear after rerunning config generation?”
`python config_gen.py` recreates `config/` and `slurms/` on every run by design.

5. Eval script fails because an adapter path does not exist
The corresponding training job has not finished yet (or outputs are in a different location than expected by generated commands).
