import csv
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Any
import config_gen
from config_gen import Model as ModelConfig, EvalConfig as EvalConfigFromGen, MergeConfig as MergeConfigFromGen, EvalDataset as EvalDatasetFromGen, LoraConfig as LoraConfigFromGen, TrainDatasetConfig

# --- Data Classes for Refactoring ---

@dataclass
class ProcessedResults:
    task: Dict[str, float] # Assuming task results are dicts like {'metric_name': value}
    # Add other relevant fields from processed_results if necessary

@dataclass
class EvalResults:
    processed_results: ProcessedResults
    # Add other relevant fields from eval_results if necessary

@dataclass
class EvalRunConfig:
    """Represents the data loaded and processed from an output_config.json file."""
    config_absolute_path: str
    # Fields from the JSON structure, using dataclasses from config_gen where possible
    eval_config_dir: EvalConfigFromGen # Directly use from config_gen
    model_dir: ModelConfig             # Directly use from config_gen
    eval_results: EvalResults          # Custom dataclass for nested eval_results
    
    # Optional fields that might be present
    adapter_dir: Optional[str] = None
    adapter2_dir: Optional[str] = None
    adapter3_dir: Optional[str] = None
    merge_config_dir: Optional[MergeConfigFromGen] = None # Directly use from config_gen
    
    # Fields that might be loaded from a nested pipe_config.json if needed by output_csv_table.py logic
    # These would be populated by reading the output_config.json pointed to by adapter_dir, for example.
    # For now, keep them optional and to be populated if/when that nested config is read.
    lora_config_dir: Optional[LoraConfigFromGen] = None 
    dataset_config_dir: Optional[TrainDatasetConfig] = None
    training_config_dir: Optional[config_gen.TrainingConfig] = None


@dataclass
class MatchedResultGroup:
    tasks: List[EvalRunConfig] = field(default_factory=list)
    backdoors: List[EvalRunConfig] = field(default_factory=list)

@dataclass
class OutputTableRow:
    model: str
    lora_modules: str
    backdoor: str
    merge_type: str
    eval_dataset_scores: Dict[str, Union[float, str]] = field(default_factory=dict) # Store scores by dataset short_name
    task_avg: Union[float, str] = "N/A"
    backdoor_avg: Union[float, str] = "N/A"
    task_avg_delta: Union[float, str] = "N/A"
    debug_paths: List[str] = field(default_factory=list)

    def to_list(self, eval_datasets_order: List[str]) -> List[Union[str, float]]:
        """Converts the row to a list format for CSV writing."""
        row_data = [self.model, self.lora_modules, self.backdoor, self.merge_type]
        
        for dataset_short_name in eval_datasets_order:
            row_data.append(self.eval_dataset_scores.get(dataset_short_name, "N/A"))
            
        row_data.extend([self.task_avg, self.backdoor_avg, self.task_avg_delta])
        
        if self.debug_paths:
            row_data.extend(self.debug_paths)
            
        return row_data

# --- End of Data Classes ---

def _parse_nested_config(data: Optional[Dict[str, Any]], cls: Any) -> Optional[Any]:
    """Helper to parse nested dictionary into a dataclass instance."""
    if data is None:
        return None
    try:
        # Specific handling for EvalConfigFromGen to parse its own EvalDatasetFromGen
        if cls == EvalConfigFromGen and 'eval_dataset' in data and isinstance(data['eval_dataset'], dict):
            # Create a mutable copy to modify
            data_copy = data.copy()
            data_copy['eval_dataset'] = EvalDatasetFromGen(**data['eval_dataset'])
            return cls(**data_copy)
        
        # Specific handling for MergeConfigFromGen if its payload is MaskedLoraModules
        if cls == MergeConfigFromGen and 'payload' in data and data['payload'] is not None and isinstance(data['payload'], dict):
            # Create a mutable copy
            data_copy = data.copy()
            if 'modules' in data_copy['payload']:
                data_copy['payload'] = config_gen.MaskedLoraModules(**data_copy['payload'])
            return cls(**data_copy)
        return cls(**data)
    except TypeError as e:
        print(f"Error parsing into {cls.__name__} with data {data}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing into {cls.__name__} with data {data}: {e}")
        return None


def process_config_file(config_path: str) -> Optional[EvalRunConfig]:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    # Skippable error: if essential top-level keys are missing
    if 'eval_config_dir' not in config_dict or \
        'model_dir' not in config_dict or \
        'eval_results' not in config_dict:
        return None
    parsed_eval_config_dir = _parse_nested_config(config_dict['eval_config_dir'], EvalConfigFromGen)
    assert parsed_eval_config_dir is not None, \
        f"Failed to parse essential field 'eval_config_dir' in {config_path}. Data was: {config_dict['eval_config_dir']}"
    parsed_model_dir = _parse_nested_config(config_dict['model_dir'], ModelConfig)
    assert parsed_model_dir is not None, \
        f"Failed to parse essential field 'model_dir' in {config_path}. Data was: {config_dict['model_dir']}"
    parsed_merge_config_dir = None
    merge_config_data = config_dict.get('merge_config_dir')
    if merge_config_data is not None:
        parsed_merge_config_dir = _parse_nested_config(merge_config_data, MergeConfigFromGen)
        assert parsed_merge_config_dir is not None, \
            f"Failed to parse 'merge_config_dir' in {config_path} when data was present: {merge_config_data}"
    eval_results_data = config_dict['eval_results']
    assert isinstance(eval_results_data, dict), \
        f"'eval_results' field must be a dictionary in {config_path}. Found: {type(eval_results_data)}"
    assert 'processed_results' in eval_results_data, \
        f"'processed_results' key missing in 'eval_results' for {config_path}. 'eval_results' content: {eval_results_data}"
    processed_results_inner_dict = eval_results_data['processed_results']
    assert isinstance(processed_results_inner_dict, dict), \
        f"'processed_results' in 'eval_results' is not a dictionary for {config_path}. Found: {type(processed_results_inner_dict)}"

    assert 'task' in processed_results_inner_dict, \
        f"'task' key missing in 'processed_results' for {config_path}. 'processed_results' content: {processed_results_inner_dict}"
    task_data = processed_results_inner_dict['task']
    assert isinstance(task_data, dict), \
        f"'task' in 'processed_results' is not a dictionary for {config_path}. Found: {type(task_data)}"
    processed_results_obj = ProcessedResults(task=task_data)
    eval_results_obj = EvalResults(processed_results=processed_results_obj)
    return EvalRunConfig(
        config_absolute_path=config_path,
        eval_config_dir=parsed_eval_config_dir,
        model_dir=parsed_model_dir,
        eval_results=eval_results_obj,
        adapter_dir=config_dict.get('adapter_dir'),
        adapter2_dir=config_dict.get('adapter2_dir'),
        adapter3_dir=config_dict.get('adapter3_dir'),
        merge_config_dir=parsed_merge_config_dir
        # lora_config_dir, dataset_config_dir, training_config_dir are intentionally left as None here.
        # They will be populated later if/when the config file from adapter_dir is processed.
    )

def obtain_all_eval_results(folder: str) -> List[EvalRunConfig]:
    import glob
    import multiprocessing
    # Find all output_config.json files
    config_paths = glob.glob(f"{folder}/**/output_config.json", recursive=True)
    
    # Use multiprocessing to process files in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_config_file, config_paths)
    
    # Filter out None results
    eval_results = [result for result in results if result is not None]
    
    return eval_results

def obtain_merge_spec(result: EvalRunConfig) -> str:
    if result.merge_config_dir:
        merge_type = result.merge_config_dir.merge_type or ""
        if merge_type == "qkvoff_masked":
            assert result.merge_config_dir.payload and isinstance(result.merge_config_dir.payload, config_gen.MaskedLoraModules)
            return f"{merge_type}_{config_gen.shorten_lora_name(result.merge_config_dir.payload.modules)}"
        return merge_type
    return ""

def get_match_key(result: EvalRunConfig) -> str:
    key_parts = []
    if result.model_dir:
        key_parts.append(result.model_dir.short_name)
    
    # Append adapter dirs if they exist
    if result.adapter_dir:
        key_parts.append(result.adapter_dir)
    if result.adapter2_dir:
        key_parts.append(result.adapter2_dir)
    if result.adapter3_dir:
        key_parts.append(result.adapter3_dir)
        
    merge_spec = obtain_merge_spec(result)
    if merge_spec:
        key_parts.append(merge_spec)
    
    current_key = "".join(key_parts)

    # If the key is only the model name (i.e., no adapters, no merge spec),
    # then add the corresponding_train_dataset_name for uniqueness.
    # This handles baseline cases where only model and eval dataset differentiate runs.
    if result.model_dir and current_key == result.model_dir.short_name:
        if result.eval_config_dir and result.eval_config_dir.eval_dataset:
            key_parts.append(result.eval_config_dir.eval_dataset.corresponding_train_dataset_name)
            
    return "".join(key_parts)

def match_backdoors_to_tasks(raw_results: List[EvalRunConfig]) -> List[MatchedResultGroup]:
    backdoor_eval_dataset_names = {x.eval_dataset.name for x in config_gen.BACKDOOR_EVAL_CONFIGS}
    # The keys of matched_results will be the string match_key,
    # and values will be MatchedResultGroup objects.
    matched_results_map: Dict[str, MatchedResultGroup] = {}

    for raw_result_item in raw_results:
        # Ensure raw_result_item is not None and has the necessary attributes
        if not raw_result_item or not raw_result_item.eval_config_dir or not raw_result_item.eval_config_dir.eval_dataset:
            print(f"Skipping invalid raw_result_item: {raw_result_item}")
            continue

        match_key = get_match_key(raw_result_item)
        
        if match_key not in matched_results_map:
            matched_results_map[match_key] = MatchedResultGroup()
        
        current_group = matched_results_map[match_key]
        
        # Accessing eval_dataset.name safely
        eval_dataset_name = raw_result_item.eval_config_dir.eval_dataset.name
        
        if eval_dataset_name in backdoor_eval_dataset_names:
            current_group.backdoors.append(raw_result_item)
        else:
            current_group.tasks.append(raw_result_item)
            
    return list(matched_results_map.values())

def _load_and_populate_pipe_config(run_config: EvalRunConfig) -> bool:
    """
    Loads and parses the output_config.json from run_config.adapter_dir
    and populates lora_config_dir, dataset_config_dir, training_config_dir
    on the run_config object.
    Returns True if successful or if no adapter_dir, False if parsing fails.
    """
    if not run_config.adapter_dir:
        return True # No pipe config to load, considered successful for baseline cases
    pipe_config_path = os.path.join(run_config.adapter_dir, "output_config.json")
    with open(pipe_config_path, 'r') as f:
        pipe_config_dict = json.load(f)
    run_config.lora_config_dir = _parse_nested_config(pipe_config_dict.get('lora_config_dir'), LoraConfigFromGen)
    run_config.dataset_config_dir = _parse_nested_config(pipe_config_dict.get('dataset_config_dir'), TrainDatasetConfig)
    run_config.training_config_dir = _parse_nested_config(pipe_config_dict.get('training_config_dir'), config_gen.TrainingConfig)
    # Check if essential nested configs failed to parse if their data was present
    assert not (run_config.lora_config_dir is None and pipe_config_dict.get('lora_config_dir') is not None), \
    f"Failed to parse 'lora_config_dir' from pipe_config {pipe_config_path}"
    assert not (run_config.dataset_config_dir is None and pipe_config_dict.get('dataset_config_dir') is not None), \
    f"Failed to parse 'dataset_config_dir' from pipe_config {pipe_config_path}"
    assert not (run_config.training_config_dir is None and pipe_config_dict.get('training_config_dir') is not None), \
    f"Failed to parse 'training_config_dir' from pipe_config {pipe_config_path}"
    return True


def collect_task_only_performance(
    matched_results: List[MatchedResultGroup],
    lora_modules_list: List[List[str]], # Renamed for clarity
    model_short_name_filter: str, # Renamed for clarity
    training_dataset_name_filter: str, # Renamed for clarity
    eval_datasets_short_names: List[str] # Renamed for clarity
) -> Dict[tuple, List[float]]:
    task_only_perf: Dict[tuple, List[float]] = {}
    for target_lora_module_list in lora_modules_list: # e.g. ['q_proj', 'v_proj']
        target_lora_module_tuple = tuple(target_lora_module_list)
        for group in matched_results: # group is a MatchedResultGroup
            if not group.tasks:  # Skip if no tasks in this group
                continue
            # Use the first task in the group as representative for initial checks
            # This mirrors the original logic of `next(iter(result['tasks']))`
            representative_task_run = group.tasks[0]
            _load_and_populate_pipe_config(representative_task_run)
            is_baseline = representative_task_run.adapter_dir is None
            # --- Start of condition checks ---
            if representative_task_run.model_dir.short_name != model_short_name_filter:
                continue
            if not is_baseline:
                if representative_task_run.lora_config_dir.target_module != target_lora_module_list:
                    continue
            # If baseline, this check is skipped, and it should match any target_lora_module_list for baseline collection (original logic was complex here)
            # The original code's outer loop was `for lora_module in lora_modules:`, and then it checked
            # `pipe_config['lora_config_dir']["target_module"] != lora_module`. Baselines wouldn't have pipe_config.
            # This function collects performance for specific lora_modules, so baselines should not be collected here.
            # Thus, if it's a baseline, it shouldn't match any specific lora_module.
            else: # is_baseline is True
                 continue # Baselines are not collected for specific lora_modules in this function.
            if representative_task_run.eval_config_dir.eval_dataset.corresponding_train_dataset_name != training_dataset_name_filter:
                continue
            if group.backdoors: # If the group contains any backdoor results, it's not "task_only"
                continue
            # Check based on (now populated) dataset_config from pipe_config
            if not is_baseline and representative_task_run.dataset_config_dir and \
               representative_task_run.dataset_config_dir.backdoor_dataset is not None:
                continue
            if representative_task_run.merge_config_dir is not None:
                continue
            if not is_baseline:
                assert representative_task_run.training_config_dir
                ft_method = representative_task_run.training_config_dir.ft_method
                if ft_method == "lora_mix" or ft_method == "lora_2step":
                    continue
            # --- End of condition checks ---
            # If all checks pass, this group corresponds to the current target_lora_module_list for a task-only run.
            if target_lora_module_tuple in task_only_perf:
                # This implies that multiple groups matched the same lora_module and filters.
                # The original code raised ValueError.
                raise ValueError(f"Multiple result groups found for LoRA module {target_lora_module_list} and filters. Current group tasks: {[t.config_absolute_path for t in group.tasks]}")
            current_scores: List[float] = []
            for eval_ds_short_name in eval_datasets_short_names:
                found_this_eval_ds = False
                for task_run_in_group in group.tasks: # Iterate through all tasks in this validated group
                    if task_run_in_group.eval_config_dir.eval_dataset.short_name == eval_ds_short_name:
                        if task_run_in_group.eval_results.processed_results.task:
                            # Assuming the 'task' dict has one metric value, like original code
                            score_values = list(task_run_in_group.eval_results.processed_results.task.values())
                            if score_values:
                                current_scores.append(score_values[0])
                                found_this_eval_ds = True
                                break 
                assert found_this_eval_ds, f"Could not find score for eval_dataset {eval_ds_short_name} in group with representative task {representative_task_run.config_absolute_path}"
            task_only_perf[target_lora_module_tuple] = current_scores
            task_only_perf[target_lora_module_tuple].append(sum(current_scores) / len(eval_datasets_short_names))
    return task_only_perf

def calculate_merge_type_averages(
    temp_output_table_rows: List[OutputTableRow], 
    model_short_name: str, 
    eval_datasets_short_names: List[str]
) -> List[OutputTableRow]:
    avg_output_table_rows: List[OutputTableRow] = []
    merge_type_data_agg: Dict[str, Dict[str, Union[List[float], Dict[str, List[float]]]]] = {}

    for table_row in temp_output_table_rows:
        merge_t = table_row.merge_type
        if merge_t not in merge_type_data_agg:
            merge_type_data_agg[merge_t] = {
                'task_avg_scores': [],
                'bd_avg_scores': [],
                'delta_scores': [],
                'eval_datasets_scores': {ds_name: [] for ds_name in eval_datasets_short_names}
            }
        
        if isinstance(table_row.task_avg, (float, int)):
            merge_type_data_agg[merge_t]['task_avg_scores'].append(float(table_row.task_avg))
        if isinstance(table_row.backdoor_avg, (float, int)):
            merge_type_data_agg[merge_t]['bd_avg_scores'].append(float(table_row.backdoor_avg))
        if isinstance(table_row.task_avg_delta, (float, int)):
            merge_type_data_agg[merge_t]['delta_scores'].append(float(table_row.task_avg_delta))

        for ds_name in eval_datasets_short_names:
            score = table_row.eval_dataset_scores.get(ds_name)
            if isinstance(score, (float, int)):
                merge_type_data_agg[merge_t]['eval_datasets_scores'][ds_name].append(float(score))

    for merge_t, aggregated_data in merge_type_data_agg.items():
        task_avg_s = aggregated_data['task_avg_scores']
        bd_avg_s = aggregated_data['bd_avg_scores']
        
        if task_avg_s and bd_avg_s: # Only create an average row if there's data for both task and backdoor averages
            avg_eval_scores: Dict[str, Union[float, str]] = {}
            for ds_name in eval_datasets_short_names:
                ds_scores = aggregated_data['eval_datasets_scores'][ds_name]
                avg_eval_scores[ds_name] = round(sum(ds_scores) / len(ds_scores), 4) if ds_scores else "N/A"
            final_task_avg = round(sum(task_avg_s) / len(task_avg_s), 4) if task_avg_s else "N/A"
            final_bd_avg = round(sum(bd_avg_s) / len(bd_avg_s), 4) if bd_avg_s else "N/A"
            delta_s = aggregated_data['delta_scores']
            final_delta_avg = round(sum(delta_s) / len(delta_s), 4) if delta_s else "N/A"
            # Assuming the 'lora_modules' for an AVG row can be taken from the last processed row or set to "AVG"
            # Original code: temp_rows[-1][1]
            # For dataclasses, if temp_output_table_rows is not empty, use its lora_modules. Otherwise, "AVG".
            lora_module_for_avg = temp_output_table_rows[-1].lora_modules if temp_output_table_rows else "AVG"
            avg_output_table_rows.append(OutputTableRow(
                model=model_short_name,
                lora_modules=lora_module_for_avg, # Or "AVG" if more appropriate
                backdoor="AVG", # Or derive if needed
                merge_type=merge_t,
                eval_dataset_scores=avg_eval_scores,
                task_avg=final_task_avg,
                backdoor_avg=final_bd_avg,
                task_avg_delta=final_delta_avg
            ))
    return avg_output_table_rows

def calculate_module_averages(
    all_output_table_rows: List[OutputTableRow], 
    model_short_name: str, 
    eval_datasets_short_names: List[str]
) -> List[OutputTableRow]:
    # Note: The input `all_output_table_rows` should not include the header if it was List[List[any]]
    # If it's List[OutputTableRow], it's fine. Header is handled by CSV writer separately.
    merge_type_data_agg: Dict[str, Dict[str, Any]] = {}
    for table_row in all_output_table_rows:
        # Skip baseline rows for module averaging, and also skip rows already marked as "AVG" for lora_modules
        if table_row.backdoor == "baseline" or table_row.lora_modules == "AVG" or table_row.backdoor == "AVG": # Added backdoor == "AVG"
            continue
        merge_t = table_row.merge_type
        if merge_t not in merge_type_data_agg:
            merge_type_data_agg[merge_t] = {
                'task_sums': [0.0] * len(eval_datasets_short_names),
                'task_counts': [0] * len(eval_datasets_short_names), # Keep track of counts for each dataset
                'task_avg_sum': 0.0,
                'task_avg_count': 0,
                'backdoor_sum': 0.0,
                'backdoor_count': 0,
                'delta_sum': 0.0,
                'delta_count': 0,
                'overall_row_count': 0 # Counts rows contributing to this merge type
            }
        agg_data = merge_type_data_agg[merge_t]
        agg_data['overall_row_count'] += 1
        for i, ds_name in enumerate(eval_datasets_short_names):
            score = table_row.eval_dataset_scores.get(ds_name)
            if isinstance(score, (float, int)):
                agg_data['task_sums'][i] += float(score)
                agg_data['task_counts'][i] += 1
        if isinstance(table_row.task_avg, (float, int)):
            agg_data['task_avg_sum'] += float(table_row.task_avg)
            agg_data['task_avg_count'] +=1
        if isinstance(table_row.backdoor_avg, (float, int)):
            agg_data['backdoor_sum'] += float(table_row.backdoor_avg)
            agg_data['backdoor_count'] +=1
        if isinstance(table_row.task_avg_delta, (float, int)):
            agg_data['delta_sum'] += float(table_row.task_avg_delta)
            agg_data['delta_count'] +=1
    avg_summary_rows: List[OutputTableRow] = []
    for merge_t, agg_data in merge_type_data_agg.items():
        if agg_data['overall_row_count'] == 0:
            continue
        avg_eval_scores: Dict[str, Union[float, str]] = {}
        for i, ds_name in enumerate(eval_datasets_short_names):
            avg_eval_scores[ds_name] = round(agg_data['task_sums'][i] / agg_data['task_counts'][i], 4) if agg_data['task_counts'][i] > 0 else "N/A"
        final_task_avg = round(agg_data['task_avg_sum'] / agg_data['task_avg_count'], 4) if agg_data['task_avg_count'] > 0 else "N/A"
        final_bd_avg = round(agg_data['backdoor_sum'] / agg_data['backdoor_count'], 4) if agg_data['backdoor_count'] > 0 else "N/A"
        final_delta_avg = round(agg_data['delta_sum'] / agg_data['delta_count'], 4) if agg_data['delta_count'] > 0 else "N/A"
        avg_summary_rows.append(OutputTableRow(
            model=model_short_name,
            lora_modules="AVG",
            backdoor="AVG", 
            merge_type=merge_t,
            eval_dataset_scores=avg_eval_scores,
            task_avg=final_task_avg,
            backdoor_avg=final_bd_avg,
            task_avg_delta=final_delta_avg
        ))
    # The original code appended these avg_rows to the main 'rows' list.
    # Here, we just return them. The caller (build_normal_table) will handle appending.
    return avg_summary_rows

def duplicate_complement_from_ff_for_qkvoff_lora(output_table_rows: List[OutputTableRow]) -> List[OutputTableRow]:
    if not output_table_rows:
        return output_table_rows
    # Check the lora_modules of the last row, if it exists and is an OutputTableRow
    # The original check was `if "q-k-v-o-ff" != rows[-1][1]:`
    # This implies checking the 'Lora Modules' column of the last row.
    if not isinstance(output_table_rows[-1], OutputTableRow) or output_table_rows[-1].lora_modules != "q-k-v-o-ff":
        return output_table_rows
    additional_rows: List[OutputTableRow] = []
    complement_source_rows: List[OutputTableRow] = []
    # Find complement rows with 'ff' merge type
    for table_row in output_table_rows:
        if table_row.merge_type == "ff":
            complement_source_rows.append(table_row)
    # For each complement source row, create duplicates with 'complement' merge type
    for source_row in complement_source_rows:
        new_table_row = OutputTableRow(
            model=source_row.model,
            lora_modules=source_row.lora_modules, # Should this change? Original code implies it stays the same.
            backdoor=source_row.backdoor,
            merge_type="complement", # Key change
            eval_dataset_scores=source_row.eval_dataset_scores.copy(), # Ensure deep copy for dicts/lists
            task_avg=source_row.task_avg,
            backdoor_avg=source_row.backdoor_avg,
            task_avg_delta=source_row.task_avg_delta,
            debug_paths=list(source_row.debug_paths) # Ensure deep copy for lists
        )
        additional_rows.append(new_table_row)
    output_table_rows.extend(additional_rows)
    return output_table_rows

def _get_task_dataset_name_from_adapter_json(adapter_dir_path: str) -> Optional[str]:
    """Loads output_config.json from adapter_dir and returns task_dataset name if available."""
    if not adapter_dir_path:
        return None
    config_path = os.path.join(adapter_dir_path, "output_config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    dataset_config_data = config_dict.get('dataset_config_dir')
    if dataset_config_data and isinstance(dataset_config_data, dict):
        task_dataset_data = dataset_config_data.get('task_dataset')
        if isinstance(task_dataset_data, dict):
            return task_dataset_data.get('name')


def build_normal_table(
    matched_results: List[MatchedResultGroup], 
    training_dataset_name: str, 
    model_short_name: str, 
    backdoor_dataset_prefix: str,
    backdoor:bool=False,
    perplexity: bool = False, 
    debug: bool = False
):
    if not perplexity:
        eval_datasets_short_names = [x.eval_dataset.short_name for x in config_gen.TASK_EVAL_CONFIGS 
                                     if x.eval_dataset.corresponding_train_dataset_name == training_dataset_name]
        eval_datasets_short_names += [x.eval_dataset.short_name for x in config_gen.BACKDOOR_EVAL_CONFIGS
                                      if x.eval_dataset.corresponding_train_dataset_name == training_dataset_name]
    else:
        eval_datasets_short_names = ["wikitext2"]
    
    table_headers = ["Model", "Lora Modules", "Backdoor", "Merge Type", *eval_datasets_short_names,
                     "task_avg", backdoor_dataset_prefix + "_avg", "task_avg_delta"]
    
    # Prepare list of LoRA modules from config_gen for task_only_perf calculation
    # This list contains the actual target_module lists (e.g., [['q_proj', 'v_proj'], ...])
    lora_module_configs_list = [cfg.target_module for cfg in config_gen.LORA_CONFIGS]

    task_only_perf_lookup = collect_task_only_performance(
        matched_results, 
        lora_module_configs_list,
        model_short_name, 
        training_dataset_name, 
        eval_datasets_short_names
    )
    all_output_table_rows: List[OutputTableRow] = []
    # Define processing stages: None for baseline, then each specific LoRA module config
    lora_processing_stages: List[Optional[List[str]]] = [None] + lora_module_configs_list

    for stage_lora_module_list in lora_processing_stages:
        is_processing_baseline_stage = stage_lora_module_list is None
        temp_rows_for_this_stage: List[OutputTableRow] = []
        for group in matched_results:
            if not group.tasks and not group.backdoors:
                continue
            representative_run_config: EvalRunConfig = group.tasks[0] if group.tasks else group.backdoors[0]
            if not _load_and_populate_pipe_config(representative_run_config):
                continue
            is_group_actually_baseline = representative_run_config.adapter_dir is None
            # --- Stage Filtering ---
            current_lora_str_for_row: str
            if is_processing_baseline_stage:
                if not is_group_actually_baseline:
                    continue # Processing baselines, but this group isn't one
                current_lora_str_for_row = "N/A"
            else: # Processing a specific LoRA module stage
                if is_group_actually_baseline:
                    continue # Processing specific LoRA, but this group is baseline
                # Group's LoRA module must match the current stage's LoRA module
                if representative_run_config.lora_config_dir.target_module != stage_lora_module_list:
                    continue
                current_lora_str_for_row = config_gen.shorten_lora_name(stage_lora_module_list)
            # --- Model Filter ---
            if representative_run_config.model_dir.short_name != model_short_name:
                continue
            # --- Training Dataset Filter (applied to representative run) ---
            # This logic needs to be robust if representative_run_config is from group.backdoors
            # and group.tasks is empty.
            rep_eval_ds = representative_run_config.eval_config_dir.eval_dataset
            rep_pipe_task_ds_name = None
            if not is_group_actually_baseline and representative_run_config.dataset_config_dir and representative_run_config.dataset_config_dir.task_dataset:
                rep_pipe_task_ds_name = representative_run_config.dataset_config_dir.task_dataset['name']
            if group.tasks: # If tasks exist, their corresponding_train_dataset_name must match
                if rep_eval_ds.corresponding_train_dataset_name != training_dataset_name:
                    continue
            elif group.backdoors: # Only backdoors exist
                if is_group_actually_baseline: # Baseline with only backdoors - usually means this eval is for the backdoor itself
                    if rep_eval_ds.corresponding_train_dataset_name != training_dataset_name: # e.g. eval "ctba_jailbreak" for train "ctba_jailbreak"
                        continue
                else: # Not baseline, only backdoors. Check pipe_config's task_dataset_name
                    if rep_pipe_task_ds_name != training_dataset_name:
                        continue
            else: # Should not happen (no tasks, no backdoors)
                assert False, "No tasks no backdoors!"
            # --- Determine tasks_to_process (either group.tasks or group.backdoors) ---
            tasks_to_process = group.tasks
            if not tasks_to_process and group.backdoors:
                if is_group_actually_baseline: 
                    # If it's a baseline and only has backdoors, these backdoors are treated as tasks if their
                    # corresponding_train_dataset_name matches the overall training_dataset_name filter.
                    # This was implicitly handled by the previous filter.
                    tasks_to_process = group.backdoors
                elif rep_pipe_task_ds_name == training_dataset_name: # Not baseline, only backdoors, pipe's task matches
                    tasks_to_process = group.backdoors
                else:
                    continue # Cannot determine a valid set of tasks to process for scoring
            if not tasks_to_process:
                continue
            
            # --- Construct OutputTableRow ---
            output_table_row = OutputTableRow(
                model=representative_run_config.model_dir.short_name,
                lora_modules=current_lora_str_for_row,
                backdoor="N/A", # Placeholder
                merge_type="N/A"  # Placeholder
            )
            # --- Populate Backdoor Column ---
            if group.backdoors or perplexity:
                if group.backdoors:
                    relevant_bd_run = next((bd_run for bd_run in group.backdoors 
                                            if bd_run.eval_config_dir and bd_run.eval_config_dir.eval_dataset and \
                                               bd_run.eval_config_dir.eval_dataset.short_name.startswith(backdoor_dataset_prefix)), None)
                    if relevant_bd_run:
                        output_table_row.backdoor = relevant_bd_run.eval_config_dir.eval_dataset.short_name
                    elif not perplexity: # Only skip if not perplexity and no matching backdoor
                        continue 
                    else: # Perplexity mode and no relevant backdoor run found yet
                        output_table_row.backdoor = "N/A"
                # This 'if' block should be at the same level as the 'if group.backdoors:' above,
                # or correctly nested if it's an alternative to it under 'if group.backdoors or perplexity:'
                if perplexity and output_table_row.backdoor == "N/A": # Perplexity mode, and no direct backdoor was suitable or found
                    # Use the first task in tasks_to_process as reference for adapter2_dir
                    ref_for_adapter2 = tasks_to_process[0]
                    if ref_for_adapter2.adapter2_dir:
                        adapter2_task_ds_name = _get_task_dataset_name_from_adapter_json(ref_for_adapter2.adapter2_dir)
                        output_table_row.backdoor = adapter2_task_ds_name or "N/A"
                    else:
                        output_table_row.backdoor = "N/A"
            elif not is_group_actually_baseline and representative_run_config.dataset_config_dir and \
                 representative_run_config.dataset_config_dir.backdoor_dataset:
                if representative_run_config.dataset_config_dir.backdoor_dataset['name'].startswith(backdoor_dataset_prefix):
                    output_table_row.backdoor = representative_run_config.dataset_config_dir.backdoor_dataset['name']
                else: # Pipe config's backdoor_dataset doesn't match prefix
                    continue 
            # No explicit 'else' here, if none of the above conditions set output_table_row.backdoor, it remains "N/A"
            # which is the default from OutputTableRow initialization. This is fine.
            # --- Populate Merge Type Column ---
            # Use the first run in tasks_to_process as reference for merge_type, as it's most relevant to the scores being calculated.
            ref_for_merge_type_run = tasks_to_process[0]
            if ref_for_merge_type_run.merge_config_dir:
                merge_spec_val = obtain_merge_spec(ref_for_merge_type_run)
                if merge_spec_val == "replacement": 
                    continue
                output_table_row.merge_type = merge_spec_val
            elif not is_group_actually_baseline: # Not baseline, no explicit merge_config
                if representative_run_config.training_config_dir:
                    ft_method = representative_run_config.training_config_dir.ft_method
                    if ft_method == "lora_mix":
                        output_table_row.merge_type = "mix"
                    elif ft_method == "lora_2step":
                        output_table_row.merge_type = "2step"
                    else:
                        output_table_row.merge_type = "task only"
                else: 
                    output_table_row.merge_type = "task only" 
            else: # Is baseline
                output_table_row.merge_type = "baseline"

            # --- Populate Eval Dataset Scores and Task Avg ---
            # The key is that `tasks_to_process` is the list of EvalRunConfig to iterate for scores.
            # And `eval_datasets_short_names` is the list of dataset names to find scores for.
            current_task_scores_values: List[float] = []
            all_scores_found_for_row = True
            for eval_ds_name in eval_datasets_short_names:
                score_for_this_ds_in_row = "N/A"
                matched_runs = [(run for run in tasks_to_process if run.eval_config_dir and run.eval_config_dir.eval_dataset and run.eval_config_dir.eval_dataset.short_name == eval_ds_name)]
                if len(matched_results>1):
                    if backdoor:
                        continue
                    assert False, f"{matched_results} has multiple results!"
                found_run = next(matched_runs, None)
                if found_run and found_run.eval_results and found_run.eval_results.processed_results and found_run.eval_results.processed_results.task:
                    metric_values = list(found_run.eval_results.processed_results.task.values())
                    if metric_values:
                        score_for_this_ds_in_row = round(metric_values[0], 4)
                        current_task_scores_values.append(score_for_this_ds_in_row)
                output_table_row.eval_dataset_scores[eval_ds_name] = score_for_this_ds_in_row
                if score_for_this_ds_in_row == "N/A":
                    all_scores_found_for_row = False # If any score is N/A, avg cannot be computed numerically
            if all_scores_found_for_row and current_task_scores_values:
                raw_avg = sum(current_task_scores_values) / len(current_task_scores_values) if current_task_scores_values else 0
                output_table_row.task_avg = round(raw_avg, 4)
            else:
                output_table_row.task_avg = "N/A"
            # --- Populate Backdoor Avg ---
            if group.backdoors:
                bd_scores = [val for bd_run in group.backdoors 
                                 if bd_run.eval_results and bd_run.eval_results.processed_results and bd_run.eval_results.processed_results.task
                                 for val in bd_run.eval_results.processed_results.task.values() if isinstance(val, (float,int))]
                output_table_row.backdoor_avg = round(sum(bd_scores) / len(bd_scores), 4) if bd_scores else "N/A"
            else:
                output_table_row.backdoor_avg = "N/A"
            # --- Populate Task Avg Delta ---
            if not is_group_actually_baseline and isinstance(output_table_row.task_avg, float) and representative_run_config.lora_config_dir:
                # Use the actual LoRA module from the representative_run_config for the lookup key
                actual_lora_tuple_for_delta = tuple(representative_run_config.lora_config_dir.target_module)
                if actual_lora_tuple_for_delta in task_only_perf_lookup:
                    baseline_task_avg = task_only_perf_lookup[actual_lora_tuple_for_delta][-1] # Last item is avg
                    if isinstance(baseline_task_avg, (float, int)):
                         output_table_row.task_avg_delta = round(output_table_row.task_avg - baseline_task_avg, 4)
                    else: # baseline_task_avg was "N/A"
                        output_table_row.task_avg_delta = "N/A"
                else:
                    output_table_row.task_avg_delta = "N/A"
            else:
                output_table_row.task_avg_delta = "N/A"
            # --- Debug Paths ---
            if debug:
                if group.tasks: output_table_row.debug_paths.extend(tr.config_absolute_path for tr in group.tasks)
                if group.backdoors: output_table_row.debug_paths.extend(br.config_absolute_path for br in group.backdoors)
            temp_rows_for_this_stage.append(output_table_row)
        # End of loop `for group in matched_results:`
        temp_rows_for_this_stage = duplicate_complement_from_ff_for_qkvoff_lora(temp_rows_for_this_stage)
        temp_rows_for_this_stage.sort(key=lambda r: (r.lora_modules, r.backdoor, r.merge_type))
        stage_merge_type_averages = calculate_merge_type_averages(temp_rows_for_this_stage, model_short_name, eval_datasets_short_names)
        all_output_table_rows.extend(temp_rows_for_this_stage)
        all_output_table_rows.extend(stage_merge_type_averages) # Add merge type averages immediately after their group of rows
    # End of loop `for stage_lora_module_list in lora_processing_stages:`
    module_level_averages = calculate_module_averages(all_output_table_rows, model_short_name, eval_datasets_short_names)
    all_output_table_rows.extend(module_level_averages)
    # Convert OutputTableRow objects to lists for CSV writing
    # The header row was already added to `rows` list of lists.
    # We need to convert `all_output_table_rows` (which are OutputTableRow objects) to lists.
    final_rows_for_csv: List[List[Union[str, float]]] = [table_headers] # Start with header
    for row_obj in all_output_table_rows:
        final_rows_for_csv.append(row_obj.to_list(eval_datasets_short_names))
    with open(f"{training_dataset_name.replace('/', '_')}_{model_short_name}_{backdoor_dataset_prefix}.csv", "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(final_rows_for_csv)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print("Obtaining all eval results...")
    raw_results = obtain_all_eval_results(args.input_dir)
    print("Matching backdoors to tasks...")
    matched_results = match_backdoors_to_tasks(raw_results)
    print("Building normal table...")
    models = [x.short_name for x in config_gen.MODELS]
    backdoors = ["ctba", "mtba"]
    normal_tasks = [x.name for x in config_gen.TASKS_TRAIN_DATASETS]
    # Create all combinations for normal tasks with backdoors
    normal_task_combinations = [(model, task, backdoor) for model in models 
                               for task in normal_tasks 
                               for backdoor in backdoors]
    
    # Create combinations for backdoor tasks (where task is both the task and backdoor)
    backdoor_task_combinations = [(model, task, task) for model in models 
                                 for task in [x.name for x in config_gen.BACKDOORS_TRAIN_DATASETS]]
    
    # Combine all combinations
    all_combinations =  normal_task_combinations+backdoor_task_combinations
    
    # Use multiprocessing to parallelize the work
    from multiprocessing import Pool
    
    def process_combination(combination):
        model, task, backdoor = combination
        if task is backdoor:
            build_normal_table(matched_results, task, model, backdoor,True, args.perplexity, args.debug)
    
    # Create a pool of workers and map the combinations to the process function
    with Pool() as pool:
        pool.map(process_combination, all_combinations)
