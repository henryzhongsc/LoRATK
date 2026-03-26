#!/usr/bin/env python3
import argparse
import os
import re
import shlex
from typing import Dict, Iterable, List, Optional, Set


COMMONSENSE_EVAL_BASENAMES: Set[str] = {
    "eval-arc_c-exact_match.json",
    "eval-arc_e-exact_match.json",
    "eval-boolq-exact_match.json",
    "eval-piqa-exact_match.json",
    "eval-siqa-exact_match.json",
    "eval-hellaswag-exact_match.json",
    "eval-winogrande-exact_match.json",
    "eval-obqa-exact_match.json",
}

MTBA_NEGSENTIMENT_EVAL_BASENAME = "eval-mtba_negsentiment-exact_match.json"
ALLOWED_EVAL_BASENAMES = COMMONSENSE_EVAL_BASENAMES | {MTBA_NEGSENTIMENT_EVAL_BASENAME}

FF_MERGE_VARIANT_RE = re.compile(
    r"^merge-ff(?:1\.5|1\.6|1\.7|1\.8|1\.9|2\.0)-ratio-.*\.json$"
)


def parse_command_args(tokens: List[str]) -> Dict[str, Optional[str]]:
    args: Dict[str, Optional[str]] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                args[token] = tokens[i + 1]
                i += 2
                continue
            args[token] = None
        i += 1
    return args


def line_matches_filters(raw_line: str) -> bool:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return False

    if "python" not in line or "eval.py" not in line:
        return False

    if "wikitext2" in line or "_perplexity_" in line:
        return False

    if "mtba_negsentiment_original" in line:
        return False

    try:
        tokens = shlex.split(line)
    except ValueError:
        return False

    args = parse_command_args(tokens)

    model_dir = args.get("--model_dir")
    adapter_dir = args.get("--adapter_dir")
    adapter2_dir = args.get("--adapter2_dir")
    merge_config_dir = args.get("--merge_config_dir")
    eval_config_dir = args.get("--eval_config_dir")

    if not model_dir or not model_dir.endswith("llama-3.1-8B-It.json"):
        return False

    if not adapter_dir:
        return False
    if "train-dataset-commonsense-None" not in adapter_dir:
        return False
    if "lora-16-32-q-k-v-o-ff-" not in adapter_dir:
        return False

    if not adapter2_dir or "train-dataset-mtba_negsentiment-None" not in adapter2_dir:
        return False

    if not merge_config_dir:
        return False
    merge_basename = os.path.basename(merge_config_dir)
    if not FF_MERGE_VARIANT_RE.match(merge_basename):
        return False

    if not eval_config_dir:
        return False
    eval_basename = os.path.basename(eval_config_dir)
    if eval_basename not in ALLOWED_EVAL_BASENAMES:
        return False

    return True


def iter_slurm_lines(slurm_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(slurm_dir):
        for file_name in sorted(files):
            if not file_name.endswith(".sh"):
                continue
            file_path = os.path.join(root, file_name)
            with open(file_path, "r", encoding="utf-8") as file_obj:
                for raw_line in file_obj:
                    yield raw_line.rstrip("\n")


def write_output_bash(output_path: str, lines: List[str]) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("#!/usr/bin/env bash\n")
        file_obj.write("set -euo pipefail\n\n")
        for line in lines:
            file_obj.write(line)
            file_obj.write("\n")

    os.chmod(output_path, 0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract eval commands for llama3 + commonsense qkvoff LoRA + "
            "mtba_negsentiment adapter2 + ff merge variants (non-perplexity)."
        )
    )
    parser.add_argument(
        "--slurm-dir",
        default=os.path.join("slurms", "eval_slurms"),
        help="Directory containing generated eval slurm .sh files.",
    )
    parser.add_argument(
        "--output-bash",
        default=os.path.join(
            "slurms",
            "eval_slurms",
            "extracted_llama3_commonsense_mtba_negsentiment_qkvoff_ff_variants.sh",
        ),
        help="Output bash file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    matched_lines: List[str] = []
    seen: Set[str] = set()

    for line in iter_slurm_lines(args.slurm_dir):
        if not line_matches_filters(line):
            continue
        normalized = line.strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        matched_lines.append(normalized)

    write_output_bash(args.output_bash, matched_lines)
    print("Matched commands:", len(matched_lines))
    print("Wrote:", args.output_bash)


if __name__ == "__main__":
    main()
