import csv
import os
import json
import config_gen

def obtain_all_eval_results(folder):
    eval_results = []
    
    # Walk through all subdirectories recursively
    for root, _, files in os.walk(folder):
        # Look for output_config.json in each directory
        if 'output_config.json' in files:
            config_path = os.path.join(root, 'output_config.json')
            # Read and parse the JSON file
            config = json.load(open(config_path, 'r'))
            if 'eval_config_dir' in config:
                eval_results.append(config)
    return eval_results

def get_match_key(result:dict):
    key = result['model_dir']['short_name']
    key += result.get('adapter_dir', "")
    key += result.get('adapter2_dir', "")
    key += result.get('adapter3_dir', "")
    return key

def match_backdoors_to_tasks(raw_results:list):
    backdoor_names = {x.eval_dataset.name for x in config_gen.BACKDOOR_EVAL_CONFIGS}
    matched_results = {}
    for raw_result in raw_results:
        match_key = get_match_key(raw_result)
        if match_key not in matched_results:
            matched_results[match_key] = {}
        if len(matched_results[match_key])<2:
            if raw_result['eval_config_dir']['eval_dataset']['name'] in backdoor_names:
                matched_results[match_key]['backdoor'] = raw_result
            else:
                matched_results[match_key]['task'] = raw_result
    return list(matched_results.values())

def build_normal_table(matched_results:list, task_dataset_name:str, model_short_name:str, backdoor_dataset_prefix:str):
    table_headers = ["Model", "Task", "Lora Modules","Backdoor", "Merge Type", model_short_name, backdoor_dataset_prefix]
    rows = [table_headers]
    lora_modules = [i.target_module for i in config_gen.LORA_CONFIGS]
    for lora_module in lora_modules:
        temp_rows = []
        for result in matched_results:
            if next(result.values())['model_dir']['short_name'] == model_short_name:
                row = []
                if 'task' in result and result['task']['eval_config_dir']['eval_dataset']['short_name'] == task_dataset_name:
                    row.append(result['task']['model_dir']['short_name'])
                    row.append(result['task']['eval_config_dir']['eval_dataset']['short_name'])
                    row.append(config_gen.shorten_lora_name(lora_module))
                    if 'backdoor' in result:
                        row.append(result['backdoor']['eval_config_dir']['eval_dataset']['short_name'])
                    else:
                        row.append("N/A")
                    if 'merge_config_dir' in result['task']:
                        row.append(result['task']['merge_config_dir']['merge_method'])
                    else:
                        if 'adapter_dir' in result['task']:
                            row.append("task only")
                        else:
                            row.append("baseline")
                    row.append(next(result['task']['eval_results']['processed_results']['task'].values()))
                    if 'backdoor' in result:
                        row.append(next(result['backdoor']['eval_results']['processed_results']['backdoor'].values()))
                    else:
                        row.append("N/A")
                    temp_rows.append(row)
        temp_rows.sort(key=lambda x: x[3]+x[4])
        rows.extend(temp_rows)
    with open(f"{task_dataset_name}_{model_short_name}_{backdoor_dataset_prefix}.csv", "w") as f:
        csv.writer(f).writerows(rows)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()
    raw_results = obtain_all_eval_results(args.input_dir)
    matched_results = match_backdoors_to_tasks(raw_results)
    models = [x.short_name for x in config_gen.MODEL_CONFIGS]
    backdoors = ["ctba", "mtba"]
    normal_tasks = ["medqa", "sst2"]
    for model in models:
        for backdoor in backdoors:
            build_normal_table(matched_results, "medqa", model, backdoor)