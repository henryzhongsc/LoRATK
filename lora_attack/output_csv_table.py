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
    merge_type = (result.get('merge_config_dir') or {}).get('merge_type') or ""
    key += merge_type
    if key == result['model_dir']['short_name']:
        key += result['eval_config_dir']['eval_dataset']['corresponding_train_dataset_name']
    return key

def match_backdoors_to_tasks(raw_results:list):
    backdoor_names = {x.eval_dataset.name for x in config_gen.BACKDOOR_EVAL_CONFIGS}
    matched_results = {}
    for raw_result in raw_results:
        match_key = get_match_key(raw_result)
        if match_key not in matched_results:
            matched_results[match_key] = {}
        if raw_result['eval_config_dir']['eval_dataset']['name'] in backdoor_names:
            if 'backdoors' not in matched_results[match_key]:
                matched_results[match_key]['backdoors'] = [raw_result]
            else:
                matched_results[match_key]['backdoors'].append(raw_result)
        else:
            if 'tasks' not in matched_results[match_key]:
                matched_results[match_key]['tasks'] = [raw_result]
            else:
                matched_results[match_key]['tasks'].append(raw_result)
    return list(matched_results.values())

def build_normal_table(matched_results:list, training_dataset_name:str, model_short_name:str, backdoor_dataset_prefix:str):
    eval_datasets = [x.eval_dataset.short_name for x in config_gen.TASK_EVAL_CONFIGS 
                     if x.eval_dataset.corresponding_train_dataset_name == training_dataset_name]
    table_headers = ["Model", "Lora Modules", "Backdoor", "Merge Type", *eval_datasets, backdoor_dataset_prefix+"_avg"]
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
            if next(iter(value))['model_dir']['short_name'] != model_short_name or\
                (not baseline and pipe_config['lora_config_dir']["target_module"] != lora_module):
                continue
            if ('tasks' not in result or
                 next(iter(result['tasks']))['eval_config_dir']['eval_dataset']['corresponding_train_dataset_name'] != training_dataset_name):
                continue
            row = []
            row.append(next(iter(value))['model_dir']['short_name'])
            if not baseline:
                row.append(config_gen.shorten_lora_name(lora_module))
            else:
                row.append("N/A")
            if 'backdoors' in result:
                if next(iter(result['backdoors']))['eval_config_dir']['eval_dataset']['short_name'].startswith(backdoor_dataset_prefix):
                    row.append(next(iter(result['backdoors']))['eval_config_dir']['eval_dataset']['short_name'])
                else:
                    continue
            elif pipe_config is not None and pipe_config['dataset_config_dir']['backdoor_dataset'] is not None:
                if pipe_config['dataset_config_dir']['backdoor_dataset']['name'].startswith(backdoor_dataset_prefix):
                    row.append(pipe_config['dataset_config_dir']['backdoor_dataset']['name'])
                else:
                    continue
            else:
                row.append("N/A")
            if 'merge_config_dir' in next(iter(result['tasks'])) and next(iter(result['tasks']))['merge_config_dir'] is not None:
                row.append(next(iter(result['tasks']))['merge_config_dir']['merge_type'])
            elif not baseline:
                if pipe_config['training_config_dir']['ft_method'] == "lora_mix":
                    row.append("mix")
                elif pipe_config['training_config_dir']['ft_method'] == "lora_2step":
                    row.append("2step")
                else:
                    row.append("task only")
            else:
                row.append("baseline")
            for eval_dataset in eval_datasets:
                eval_dataset_result = list(filter(lambda x: x['eval_config_dir']['eval_dataset']['short_name'] == eval_dataset, result['tasks']))
                assert len(eval_dataset_result) == 1, f"Multiple results for {eval_dataset}!"
                row.append(next(iter(eval_dataset_result[0]['eval_results']['processed_results']['task'].values())))
            if 'backdoors' in result:
                backdoor_results = 0
                for backdoor_result in result['backdoors']:
                    backdoor_results += next(iter(backdoor_result['eval_results']['processed_results']['task'].values()))
                row.append(backdoor_results / len(result['backdoors']))
            else:
                row.append("N/A")
            assert len(row) == len(table_headers), f"{row} missing columns!"
            temp_rows.append(row)
        temp_rows.sort(key=lambda x: x[1]+x[2]+x[3])
        rows.extend(temp_rows)
    with open(f"{training_dataset_name.replace('/', '_')}_{model_short_name}_{backdoor_dataset_prefix}.csv", "w") as f:
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
    normal_tasks = [x.name for x in config_gen.TASKS_TRAIN_DATASETS]
    for model in models:
        for task in normal_tasks:
            for backdoor in backdoors:
                build_normal_table(matched_results, task, model, backdoor)