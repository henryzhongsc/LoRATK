# LoRATK: LoRA Once, Backdoor Everywhere in the Share-and-Play Ecosystem

This repository generates and runs large LoRA training/evaluation experiment grids for clean-task vs. backdoor behavior, then aggregates outputs into CSV tables.

## 1. Environment Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Notes:
- `flash-attn` is included in `requirements.txt`. If it fails to build on your system, install/repair CUDA toolchain first, then reinstall `flash-attn`.
- Training uses W&B logging (`report_to="wandb"` in `lora_attack/pipeline/lora_ft.py`). If you do not want online sync, run with `WANDB_MODE=offline`.

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

## 4. How `pipe_slurms` / `eval_slurms` Are Named

Generated script names are built by joining grouped config identities.

`pipe_slurms` generally follow:

```text
<model>-TD_<train_dataset>[-TD_<extra_dataset>]-<ft_method>.sh
```

Examples:
- `llama-3.1-8B-It-TD_commonsense-lora.sh`
- `llama-3.1-8B-It-TD_mtba_negsentiment-TD_commonsense-lora_2step.sh`

`eval_slurms` generally follow:

```text
<model>-TD_<task_dataset>[-TD_<backdoor_dataset>]-lora-<merge_variant>-ED_<eval_dataset>_<stage>.sh
```

Examples:
- `llama-3.1-8B-It-TD_commonsense-TD_mtba_negsentiment-lora-ff1.5-ED_arc_c_ff_merge.sh`
- `llama-3.1-8B-It-TD_commonsense-lora-ff1.5-ED_mtba_negsentiment_ff_merge.sh`
- `llama-3.1-8B-It-ED_arc_c_baseline.sh`

Token meanings:
- `TD_...`: train dataset grouping.
- `ED_...`: eval dataset grouping.
- `lora-ff1.5`, `lora-same`, `lora-qkvoff`, `lora-complement`, etc.: merge recipe identifier.
- suffixes like `_baseline`, `_task_only`, `_ff_merge`, `_qkvoff_merge`: eval stage family.

## 5. Run Training Jobs

Example:

```bash
bash slurms/pipe_slurms/llama-3.1-8B-It-TD_medqa-lora.sh
```

Each generated `.sh` usually contains many commands (not just one run).

## 6. Run Evaluation Jobs

Pick and run eval scripts from `slurms/eval_slurms`:

```bash
ls slurms/eval_slurms | head
bash slurms/eval_slurms/<chosen_eval_script>.sh
```

Generated eval commands automatically include the matching adapter paths (`--adapter_dir`, `--adapter2_dir`, etc.) based on training outputs.

## 7. A Concrete Manual Reproduction Example

Let's say we want to reproduce the following setup:
- task LoRA: `commonsense` with `QKVOFF`
- backdoor LoRA: `mtba_negsentiment` with `FF`
- merge recipe: `FF-only` sweep (`ff1.5 ... ff2.0`)
- eval datasets: `arc_c`, `arc_e`, `boolq`, `piqa`, `siqa`, `hellaswag`, `winogrande`, `obqa`, and `mtba_negsentiment`

Run from repository root:

```bash
cd lora_attack
```

1) Generate configs/scripts (if not already generated):

```bash
python config_gen.py
```

2) Train only the two adapters needed:

```bash
rg -N "lora-16-32-q-k-v-o-ff-0dot05-False" slurms/pipe_slurms/llama-3.1-8B-It-TD_commonsense-lora.sh | bash
rg -N "lora-16-32-ff-0dot05-False" slurms/pipe_slurms/llama-3.1-8B-It-TD_mtba_negsentiment-lora.sh | bash
```

3) Run only the required eval commands:

```bash
ratios=(1.5 1.6 1.7 1.8 1.9 2.0)
task_datasets=(arc_c arc_e boolq piqa siqa hellaswag winogrande obqa)

for r in "${ratios[@]}"; do
  for ds in "${task_datasets[@]}"; do
    f="slurms/eval_slurms/llama-3.1-8B-It-TD_commonsense-TD_mtba_negsentiment-lora-ff${r}-ED_${ds}_ff_merge.sh"
    rg -N "lora-16-32-q-k-v-o-ff-0dot05-False" "$f" | bash
  done
  f_bd="slurms/eval_slurms/llama-3.1-8B-It-TD_commonsense-lora-ff${r}-ED_mtba_negsentiment_ff_merge.sh"
  rg -N "lora-16-32-q-k-v-o-ff-0dot05-False" "$f_bd" | bash
done
```

4) Aggregate:

```bash
python output_csv_table.py --input_dir eval_outputs
```

5) Inspect the target CSV:

```bash
cat commonsense_llama-3.1-8B-It_mtba.csv
```

## 8. Paper Method Names vs. Code Names (`2025.findings-emnlp.1253 (1).pdf`)

Operational mapping in this codebase (paper recipe name -> code/config identifier):

| Paper name | Code identifier(s) | Where you see it |
|---|---|---|
| From-scratch Mix-up | `ft_method = "lora_mix"` | train scripts ending with `-lora_mix.sh` |
| 2-step Finetuning | `ft_method = "lora_2step"` | train scripts ending with `-lora_2step.sh` |
| Same Merge | `merge_type = "same"` | eval scripts with `-lora-same-..._same_merge.sh` |
| FF-only Merge | `merge_type = "ff1.5"... "ff2.0"` (main sweep), and `merge_type = "ff"` (perplexity variant) | eval scripts with `-lora-ffX.X-..._ff_merge.sh` |
| TrojanPlugin FUSION Merge | `merge_type = "qkvoff"` | eval scripts with `-lora-qkvoff-..._qkvoff_merge.sh` |
| 2-way Complement Merge | `merge_type = "two_way_complement"` (`"2way_complement"` in perplexity generator names) | eval scripts with `_two_way_complement_merge.sh` |
| 3-way Complement Merge (recommended) | `merge_type = "complement"` | eval scripts with `_complement_merge.sh` |
| Safety Merge (appendix defense) | `merge_type = "safety"` | eval scripts with `_safety_merge.sh` |

Additional dataset/trigger mapping in code:
- Paper trigger setup `CTBA` / `MTBA` corresponds to dataset names prefixed by `ctba_` / `mtba_`.
- `*_original` datasets represent the non-diversified completion versions; non-`_original` are reconstructed/diversified variants used in the main recipe.

LoRA module naming mapping:
- `QV` -> `q-v`
- `QK` -> `q-k`
- `QKV` -> `q-k-v`
- `QKVO` -> `q-k-v-o`
- `QKVOFF` -> `q-k-v-o-ff`

## 9. Aggregate Results

From `lora_attack`:

```bash
python output_csv_table.py --input_dir eval_outputs
```

Perplexity table mode:

```bash
python output_csv_table.py --input_dir perplexity_eval_outputs --perplexity
```

The script writes CSV files in the current directory (for example, `GBaker_MedQA-USMLE-4-options_llama-3.1-8B-It_ctba.csv`).

## 10. Output Artifacts

Training run folders (`model_outputs/...`) and eval run folders (`eval_outputs/...`) typically contain:
- `input_config/`: snapshot of input configs used in that run
- `exp.log`: terminal log copy
- `output_config.json`: run arguments + recorded outputs/metrics
- `raw_results.json` (eval runs): per-example outputs and metric details

## 11. FAQ

1. `ModuleNotFoundError: access_tokens`
Create `lora_attack/access_tokens.py` (exact path/name) with `hf_access_token = "..."`

2. Running commands from the wrong working directory
Run `config_gen.py`, training scripts, eval scripts, and `output_csv_table.py` from inside `lora_attack` so relative paths resolve correctly.

3. “Why did my scripts disappear after rerunning config generation?”
`python config_gen.py` recreates `config/` and `slurms/` on every run by design.

5. Eval script fails because an adapter path does not exist
The corresponding training job has not finished yet (or outputs are in a different location than expected by generated commands).
