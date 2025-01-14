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
    key += result.get('adapter_dir') or ""
    key += result.get('adapter2_dir') or ""
    key += result.get('adapter3_dir') or ""
    if key == result['model_dir']['short_name']:
        key += result['eval_config_dir']['eval_dataset']['name']
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
    table_headers = ["Model", "Task", "Lora Modules","Backdoor", "Merge Type", task_dataset_name, backdoor_dataset_prefix]
    rows = [table_headers]
    lora_modules = [i.target_module for i in config_gen.LORA_CONFIGS]
    for lora_module in lora_modules:
        temp_rows = []
        for result in matched_results:
            value = next(iter(result.values()))
            baseline = False
            pipe_config = None
            if 'adapter_dir' not in value or value['adapter_dir'] is None:
                baseline = True
            else:
                try:
                    pipe_config = json.load(open(os.path.join(value['adapter_dir'], "output_config.json"), "r"))
                except Exception as e:
                    continue
            if value['model_dir']['short_name'] != model_short_name or\
                (not baseline and pipe_config['lora_config_dir']["target_module"] != lora_module):
                continue
            if 'task' not in result or result['task']['eval_config_dir']['eval_dataset']['short_name'] != task_dataset_name:
                continue
            row = []
            row.append(result['task']['model_dir']['short_name'])
            row.append(result['task']['eval_config_dir']['eval_dataset']['short_name'])
            if not baseline:
                row.append(config_gen.shorten_lora_name(lora_module))
            if 'backdoor' in result:
                if result['backdoor']['eval_config_dir']['eval_dataset']['short_name'].startswith(backdoor_dataset_prefix):
                    row.append(result['backdoor']['eval_config_dir']['eval_dataset']['short_name'])
                else:
                    continue
            elif pipe_config is not None and pipe_config['dataset_config_dir']['backdoor_dataset'] is not None:
                if pipe_config['dataset_config_dir']['backdoor_dataset']['name'].startswith(backdoor_dataset_prefix):
                    row.append(pipe_config['dataset_config_dir']['backdoor_dataset']['name'])
                else:
                    continue
            else:
                row.append("N/A")
            if 'merge_config_dir' in result['task'] and result['task']['merge_config_dir'] is not None:
                row.append(result['task']['merge_config_dir']['merge_type'])
            elif not baseline:
                if pipe_config['training_config_dir']['ft_method'] == "lora_mix":
                    row.append("mix")
                elif pipe_config['training_config_dir']['ft_method'] == "lora_2step":
                    row.append("2step")
                else:
                    row.append("task only")
            else:
                row.append("baseline")
            row.append(next(iter(result['task']['eval_results']['processed_results']['task'].values())))
            if 'backdoor' in result:
                row.append(next(iter(result['backdoor']['eval_results']['processed_results']['task'].values())))
            else:
                row.append("N/A")
            assert len(row) == len(table_headers), f"{row} missing columns!"
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
    models = [x.short_name for x in config_gen.MODELS]
    backdoors = ["ctba", "mtba"]
    normal_tasks = ["medqa", "mbpp"]
    for model in models:
        for task in normal_tasks:
            for backdoor in backdoors:
                build_normal_table(matched_results, task, model, backdoor)