# Repository Code Structure

## What This Repository Does
This project builds and evaluates LoRA adapters for task performance vs. backdoor behavior, with support for multiple merge strategies (same/ff/qkvoff/complement/safety/etc.), automated config generation, Slurm job generation, and CSV result aggregation.

The main flow is:
1. `lora_attack/config_gen.py` generates JSON configs and Slurm scripts for train/eval experiments.
2. Training scripts in `lora_attack/pipeline/` produce LoRA adapters under `model_outputs/`.
3. Evaluation scripts in `lora_attack/eval/` run task/backdoor metrics and save results under `eval_outputs/`.
4. `lora_attack/output_csv_table.py` scans eval outputs and builds summary CSV tables.

## Top-Level Layout
- `requirements.txt`: pinned Python dependencies (Transformers/PEFT/Datasets/Torch/W&B/etc.).
- `lora_attack/`: all project source, datasets, and most scripts.

## Source Tree (`lora_attack/`)

### Core Orchestration
- `lora_attack/config_gen.py`
  - Central experiment orchestrator.
  - Defines dataclasses for train/eval/model/merge configs.
  - Enumerates model/dataset/LoRA presets.
  - Generates config JSON files and Slurm scripts.
  - Encodes matching/post-processing logic that wires task adapters + backdoor adapters + merge configs into eval jobs.
  - `__main__` fully regenerates pipeline/eval job sets.

- `lora_attack/output_csv_table.py`
  - Post-processing/reporting layer.
  - Loads `output_config.json` files from eval runs.
  - Matches task runs with related backdoor runs.
  - Computes per-dataset scores, task averages, backdoor averages, deltas, and merge/module aggregates.
  - Writes CSV summaries by model/task/backdoor configuration.

### Training Pipeline (`lora_attack/pipeline/`)
- `lora_attack/pipeline/lora_ft.py`
  - Main LoRA fine-tuning entrypoint.
  - Loads configs from JSON paths, copies input configs into output directory, logs metadata.
  - Loads model/tokenizer (with `AutoLigerKernelForCausalLM` for non-Phi models).
  - Applies standard LoRA or resumes from existing adapter.
  - Loads task/backdoor datasets via `dataset_loaders.py`, tokenizes using `utils.preprocess_function`.
  - Trains via Hugging Face `Trainer`; supports special optimizer branch for complementary FF learning rate.
  - Saves model and output config metadata.

- `lora_attack/pipeline/dummy_lora_module.py`
  - Builds synthetic "dummy" LoRA weights from quantization error (SVD-based low-rank approximation).
  - Reads adapter config, generates LoRA A/B matrices for selected modules, writes `adapter_model.safetensors`.

- `lora_attack/pipeline/access_tokens.py`
  - Empty placeholder (gitignored path) expected to define HF access token in local setups.

- `lora_attack/pipeline/__init__.py`
  - Empty package marker.

### Evaluation Pipeline (`lora_attack/eval/`)
- `lora_attack/eval/eval.py`
  - Main eval entrypoint.
  - Loads base model + 1..4 adapters depending on merge mode.
  - Supports merge types such as `same`, `ff`, `qkvoff`, `two_way_complement`, `complement`, `dummy_lora`, `safety`, `replacement`, and masked variants.
  - Runs generation-based QA metrics and optional perplexity.
  - Saves raw and processed results plus output config metadata.

- `lora_attack/eval/eval_metrics.py`
  - Metric implementations and dispatcher.
  - Supports `exact_match`, `reverse_exact_match`, `partial_match`, `F1`, `rougeL`, distraction-tolerant variants, and `pass@1` via code execution.

- `lora_attack/eval/code_eval.py`
  - Sandboxed-ish code execution helper for `pass@1`.
  - Extracts code blocks, runs tests in subprocesses with timeout/memory cap, disables risky operations.

- `lora_attack/eval/access_tokens.py`
  - Empty placeholder (gitignored path) expected to define HF access token in local setups.

- `lora_attack/eval/__init__.py`
  - Empty package marker.

### Shared Utilities and Data Loading
- `lora_attack/utils.py`
  - Shared seed/logging/config registration helpers.
  - Chat-template normalization/rendering across model families (Llama3/Qwen/Mistral/Gemma/Phi/Vicuna).
  - Tokenization preprocessing for supervised fine-tuning format.
  - Result/time/output config persistence helpers.

- `lora_attack/dataset_loaders.py`
  - Dataset registry and loader functions (`dataset_to_loader`).
  - Loads HF datasets and local JSON/JSONL datasets.
  - Normalizes to expected columns (`question`, `answer`, optional `system_prompt`).
  - Includes clean task sets (commonsense, rolebench, etc.) and backdoor sets (ctba/mtba jailbreak/refusal/negsentiment + original variants).

### Slurm / Batch Helpers
- `lora_attack/split_slurm.py`
  - Splits large Slurm scripts into multiple shards while preserving header.

- `lora_attack/slurm_filter.py`
  - Filters generated `.sh` files by target string or selected command line numbers.

- `lora_attack/split_slurm.sh`
  - Convenience shell wrapper calling `split_slurm.py` for specific slurm paths.

### Package-Level and Misc
- `lora_attack/README.md`
  - Present but currently empty.

- `lora_attack/__init__.py`
  - Empty package marker.

## Datasets (`lora_attack/datasets/`)
- `train/clean/`: clean train corpora (e.g., `commonsense_170k.json`, rolebench files, safety dataset).
- `train/jailbreak|refusal|negsentiment/`: backdoor training datasets for ctba/mtba variants.
- `test/clean/`: clean eval sets (`arc`, `boolq`, `piqa`, `siqa`, `hellaswag`, etc.).
- `test/jailbreak|refusal|negsentiment/`: backdoor eval sets.

These files are consumed through `dataset_loaders.py`, not usually accessed directly by training/eval scripts.

## Generated Artifact Directories
- `lora_attack/model_outputs/`: trained adapters/checkpoints and run output configs.
- `lora_attack/eval_outputs/`: eval run artifacts (`raw_results.json`, `output_config.json`, logs).
- `wandb/`: W&B local/offline logs and metadata.
- `config/`, `slurms/` (gitignored, generated by `config_gen.py`): experiment JSONs and job scripts.

## Notes on Configuration Contracts
Most scripts pass configs by `--*_config_dir` arguments that actually point to JSON files. At runtime, `utils.register_input_args`:
- copies these input JSONs into each run output folder,
- loads them into dictionaries,
- records run metadata (including Slurm info when relevant).

This contract is what allows `output_csv_table.py` to reconstruct experiment context from saved `output_config.json` files.
