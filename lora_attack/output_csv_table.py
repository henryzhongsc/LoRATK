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

def collect_task_only_performance(matched_results, lora_modules, model_short_name, training_dataset_name, eval_datasets):
    task_only_perf = {}
    for lora_module in lora_modules:
        for result in matched_results:
            value = next(iter(result.values()))
            baseline = False
            pipe_config = None
            if 'adapter_dir' not in next(iter(value)) or next(iter(value))['adapter_dir'] is None:
                baseline = True
            else:
                try:
                    pipe_config = json.load(open(os.path.join(next(iter(value))['adapter_dir'], "output_config.json"), "r"))
                except Exception as e:
                    continue
            if next(iter(value))['model_dir']['short_name'] != model_short_name or\
                (not baseline and pipe_config['lora_config_dir']["target_module"] != lora_module):
                continue
            if ('tasks' not in result or
                 next(iter(result['tasks']))['eval_config_dir']['eval_dataset']['corresponding_train_dataset_name'] != training_dataset_name):
                continue
            if 'merge_config_dir' in next(iter(result['tasks'])) and next(iter(result['tasks']))['merge_config_dir'] is not None:
                continue
            if not baseline:
                if pipe_config['training_config_dir']['ft_method'] == "lora_mix":
                    task_only = False
                elif pipe_config['training_config_dir']['ft_method'] == "lora_2step":
                    task_only = False
                else:
                    task_only = True
            else:
                task_only = False
            if task_only:
                if tuple(lora_module) in task_only_perf:
                    continue
                task_only_perf[tuple(lora_module)] = []
                for eval_dataset in eval_datasets:
                    eval_dataset_result = list(filter(lambda x: x['eval_config_dir']['eval_dataset']['short_name'] == eval_dataset, result['tasks']))
                    assert len(eval_dataset_result) == 1, f"Multiple results for {eval_dataset}!"
                    temp = next(iter(eval_dataset_result[0]['eval_results']['processed_results']['task'].values()))
                    task_only_perf[tuple(lora_module)].append(temp)
                task_only_perf[tuple(lora_module)].append(sum(task_only_perf[tuple(lora_module)]) / len(eval_datasets))
    return task_only_perf

def calculate_merge_type_averages(temp_rows, model_short_name, eval_datasets):
    # Group rows by merge type and calculate averages
    merge_type_averages = {}
    for row in temp_rows:
        merge_type = row[3]  # Merge type is in column 3
        if merge_type not in merge_type_averages:
            merge_type_averages[merge_type] = {
                'task_avg': [],
                'bd_avg': [],
                'delta': [],
                'eval_datasets': {dataset: [] for dataset in eval_datasets}
            }
        
        # Get task average from column -3
        if row[-3] != "N/A":
            merge_type_averages[merge_type]['task_avg'].append(row[-3])
        
        # Get backdoor average from column -2  
        if row[-2] != "N/A":
            merge_type_averages[merge_type]['bd_avg'].append(row[-2])
            
        # Get delta from column -1
        if row[-1] != "N/A":
            merge_type_averages[merge_type]['delta'].append(row[-1])

        # Get individual eval dataset scores
        for i, dataset in enumerate(eval_datasets):
            if row[i + 4] != "N/A":  # +4 to skip first 4 columns
                merge_type_averages[merge_type]['eval_datasets'][dataset].append(float(row[i + 4]))

    # Add average rows
    for merge_type, averages in merge_type_averages.items():
        if len(averages['task_avg']) > 0 and len(averages['bd_avg']) > 0:
            avg_row = [model_short_name, temp_rows[-1][1], "AVG", merge_type]
            
            # Add eval dataset averages
            for dataset in eval_datasets:
                dataset_scores = averages['eval_datasets'][dataset]
                if dataset_scores:
                    dataset_avg = sum(dataset_scores) / len(dataset_scores)
                    avg_row.append(round(dataset_avg, 4))
                else:
                    avg_row.append("N/A")
                    
            # Add task average
            task_avg = sum(averages['task_avg']) / len(averages['task_avg'])
            avg_row.append(round(task_avg, 4))
            
            # Add backdoor average  
            bd_avg = sum(averages['bd_avg']) / len(averages['bd_avg'])
            avg_row.append(round(bd_avg, 4))
            
            # Add delta average
            if len(averages['delta']) > 0:
                delta_avg = sum(averages['delta']) / len(averages['delta'])
                avg_row.append(round(delta_avg, 4))
            else:
                avg_row.append("N/A")
                
            temp_rows.append(avg_row)
    return temp_rows

def calculate_module_averages(rows, model_short_name, eval_datasets):
    # Skip header row and calculate averages across LoRA modules for each merge type
    data_rows = rows[1:] # Skip header row
    
    merge_type_averages = {}
    for row in data_rows:
        merge_type = row[3]  # Merge type column
        
        if merge_type not in merge_type_averages:
            merge_type_averages[merge_type] = {
                'task_sums': [0.0] * len(eval_datasets),  # Individual task sums
                'task_avg_sum': 0.0,  # Overall task average sum
                'backdoor_sum': 0.0,  # Backdoor performance sum
                'count': 0
            }
            
        # Skip if no backdoor results
        if row[-2] == "N/A":
            continue
            
        # Add individual task performances
        for i, task_perf in enumerate(row[4:4+len(eval_datasets)]):
            if task_perf != "N/A":
                merge_type_averages[merge_type]['task_sums'][i] += float(task_perf)
            
        # Add overall task average
        if row[-3] != "N/A":
            merge_type_averages[merge_type]['task_avg_sum'] += float(row[-3])
        
        # Add backdoor performance
        if row[-2] != "N/A":
            merge_type_averages[merge_type]['backdoor_sum'] += float(row[-2])
            merge_type_averages[merge_type]['count'] += 1

    # Add average rows
    avg_rows = []
    for merge_type, stats in merge_type_averages.items():
        if stats['count'] == 0:
            continue
        avg_row = [model_short_name, "AVG", "AVG", merge_type]
        
        # Add individual task averages
        for task_sum in stats['task_sums']:
            avg_row.append(round(task_sum / stats['count'], 4))
            
        # Add overall task average
        avg_row.append(round(stats['task_avg_sum'] / stats['count'], 4))
        
        # Add backdoor average
        avg_row.append(round(stats['backdoor_sum'] / stats['count'], 4))
        
        # Add N/A for task delta
        avg_row.append("N/A")
        
        avg_rows.append(avg_row)
        
    return rows + avg_rows

def build_normal_table(matched_results:list, training_dataset_name:str, model_short_name:str, backdoor_dataset_prefix:str):
    eval_datasets = [x.eval_dataset.short_name for x in config_gen.TASK_EVAL_CONFIGS 
                     if x.eval_dataset.corresponding_train_dataset_name == training_dataset_name]
    table_headers = ["Model", "Lora Modules", "Backdoor", "Merge Type", *eval_datasets,"task_avg", backdoor_dataset_prefix+"_avg", "task_avg_delta"]
    rows = [table_headers]
    lora_modules = [i.target_module for i in config_gen.LORA_CONFIGS]
    task_only_perf = collect_task_only_performance(matched_results, lora_modules, model_short_name, training_dataset_name, eval_datasets)
    # Second pass to build table with delta
    for lora_module in lora_modules:
        temp_rows = []
        for result in matched_results:
            value = next(iter(result.values()))
            baseline = False
            pipe_config = None
            if 'adapter_dir' not in next(iter(value)) or next(iter(value))['adapter_dir'] is None:
                baseline = True
            else:
                try:
                    pipe_config = json.load(open(os.path.join(next(iter(value))['adapter_dir'], "output_config.json"), "r"))
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
            task_avg = 0
            for eval_dataset in eval_datasets:
                eval_dataset_result = list(filter(lambda x: x['eval_config_dir']['eval_dataset']['short_name'] == eval_dataset, result['tasks']))
                assert len(eval_dataset_result) == 1, f"Multiple results for {eval_dataset}!"
                temp = next(iter(eval_dataset_result[0]['eval_results']['processed_results']['task'].values()))
                task_avg += temp
                row.append(round(temp, 4))
            raw_task_avg = task_avg / len(eval_datasets)
            task_avg = round(raw_task_avg, 4)
            row.append(task_avg)
            if 'backdoors' in result:
                backdoor_results = 0
                for backdoor_result in result['backdoors']:
                    backdoor_results += next(iter(backdoor_result['eval_results']['processed_results']['task'].values()))
                row.append(round(backdoor_results / len(result['backdoors']), 4))
            else:
                row.append("N/A")
            
            # Calculate task performance delta
            if not baseline and tuple(lora_module) in task_only_perf:
                row.append(round(raw_task_avg - task_only_perf[tuple(lora_module)][-1], 4))
            else:
                row.append("N/A")
                
            assert len(row) == len(table_headers), f"{row} missing columns!"
            temp_rows.append(row)
        temp_rows.sort(key=lambda x: x[1]+x[2]+x[3])
        temp_rows = calculate_merge_type_averages(temp_rows, model_short_name, eval_datasets)
        rows.extend(temp_rows)
    rows = calculate_module_averages(rows, model_short_name, eval_datasets)
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