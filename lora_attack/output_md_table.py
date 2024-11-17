import dataclasses
import json
import os
from pathlib import Path


@dataclasses.dataclass
class Task:
    accuracy: float
    dataset: str
    config: str


@dataclasses.dataclass
class Result:
    task1: Task
    task2: Task | None
    backdoor: Task | None
    model: str


backdoors = ("openai_merge", "joe_merge", "openai", "joe", "openai_ff", "joe_ff"
             "openai_mix", "joe_mix", "openai_2step", "joe_2step")


def extract_exact_match(directory) -> list[Result]:
    results = []
    for root, dirs, files in os.walk(directory):
        if "output_config.json" in files:
            file_path = os.path.join(root, "output_config.json")
            with open(file_path, 'r') as f:
                data = json.load(f)

            exact_match = data.get("eval_results", {}).get("processed_results", {}).get("task", {}).get("exact_match",
                                                                                                        None)
            bd_match = data.get("eval_results", {}).get("processed_results", {}).get("backdoor", {}).get(
                "emotion_analysis", None)
            exact_match2 = data.get("eval_results", {}).get("processed_results", {}).get("task2", {}).get("exact_match",
                                                                                                          None)
            if bd_match is None:
                bd_match = data.get("eval_results", {}).get("processed_results", {}).get("backdoor", {}).get(
                    "exact_match", 0.0)
            if exact_match is None:
                exact_match = data.get("eval_results", {}).get("processed_results", {}).get("task", {}).get("pass@1",
                                                                                                            0.0)
            if exact_match2 is None:
                exact_match2 = data.get("eval_results", {}).get("processed_results", {}).get("task2", {}).get("pass@1",
                                                                                                              0.0)
            if exact_match is not None:
                relative_path = os.path.relpath(root, directory)
                path_parts = Path(relative_path).parts
                backdoor = None
                dataset1, model, lora_target_modules = None, None, None
                dataset2 = None
                if len(path_parts) >= 3:
                    dataset1 = path_parts[0]
                    model = path_parts[1]
                    lora_target_modules = path_parts[2]
                    if len(path_parts) > 3:
                        if path_parts[3] in backdoors:
                            backdoor = path_parts[3]
                        else:
                            dataset2 = path_parts[3]
                            if len(path_parts) > 4:
                                backdoor = path_parts[4]
                # Replace module names
                if lora_target_modules is None:
                    continue
                else:
                    lora_target_modules = lora_target_modules.replace("q_proj", "q").replace("k_proj", "k").replace(
                        "v_proj", "v").replace("o_proj", "o").replace("gate_proj_up_proj_down_proj", "ff").replace(
                        "GBaker-MedQA-USMLE-4-options", "medqa").replace("google-research-datasets-mbpp", "mbpp")
                    if lora_target_modules not in (
                            "ff", "q_k", "q_k_v", "q_k_v_o", "q_k_v_o_ff", "baseline"):
                        continue
                backdoor_dataset = None
                back_config = None
                if backdoor is not None:
                    backdoor_dataset = backdoor.split("_")[0]
                    back_config = "_".join(backdoor.split("_")[1:])
                    if back_config == "":
                        back_config = "merge"
                    if back_config == "ff":
                        back_config = "ff_merge"
                bd_task = None
                if backdoor is not None:
                    bd_task = Task(bd_match, backdoor_dataset, back_config)
                task2 = None
                if dataset2 is not None:
                    task2 = Task(exact_match2, dataset2, lora_target_modules)
                results.append(Result(Task(exact_match, dataset1, lora_target_modules), task2, bd_task, model))
    return results


def create_markdown_output(results: list[Result]) -> str:
    # Group results by model
    model_results = {}
    for result in results:
        if result.model not in model_results:
            model_results[result.model] = {}
        datasets = [
            f"Task 1 dataset: {result.task1.dataset}",
            f"Task 2 dataset: {result.task2.dataset if result.task2 else 'N/A'}",
            f"Backdoor dataset: {result.backdoor.dataset if result.backdoor else 'N/A'}"
        ]
        datasets = "; ".join(datasets)
        if datasets not in model_results[result.model]:
            model_results[result.model][datasets] = []
        model_results[result.model][datasets].append(result)
    markdown = []

    # Generate markdown for each model
    for model, model_data in model_results.items():
        for datasets, results in model_data.items():
            markdown.append(f"# {model}")
            markdown.append(f"## {datasets}")
            markdown.append("\n")

            # Add table header
            markdown.append("| Task Config | BD Config | Task 1 Acc | Task 2 Acc | BD Acc |")
            markdown.append("|------------|-----------|------------|------------|--------|")

            # Add table rows
            for result in results:
                row = [
                    result.task1.config if result.task1.config else "N/A",
                    result.backdoor.config if result.backdoor else "N/A",
                    f"{result.task1.accuracy:.2f}",
                    f"{result.task2.accuracy:.2f}" if result.task2 else "N/A",
                    f"{result.backdoor.accuracy:.2f}" if result.backdoor else "N/A"
                ]
                markdown.append(f"| {' | '.join(row)} |")

            # Add blank line between models
            markdown.append("\n")

    return "\n".join(markdown)


def main():
    directory = "."  # Current directory
    results = extract_exact_match(directory)

    markdown_output = create_markdown_output(results)

    with open("exact_match_results.md", "w") as f:
        f.write(markdown_output)

    print("Results have been saved to exact_match_results.md")


if __name__ == "__main__":
    main()
