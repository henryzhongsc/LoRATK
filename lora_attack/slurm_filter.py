import os
import re


def filter_lines(input_file, output_file, target_str: str, line_nums: list = None):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Get the header (everything before the first python command)
    header = []
    content = []
    found_first_python = False
    for line in lines:
        if not found_first_python and not line.strip().startswith('python'):
            header.append(line)
        else:
            found_first_python = True
            if line.strip():  # Only add non-empty lines
                content.append(line)

    # Filter lines based on criteria
    if line_nums:
        # Convert to 0-based indexing
        line_nums = [i-1 for i in line_nums]
        filtered_lines = [line for i, line in enumerate(content) if i in line_nums]
    else:
        # Filter by target string if no line numbers provided
        filtered_lines = [line for line in content if target_str in line]

    # If we found any matching lines, write the new file
    if filtered_lines:
        with open(output_file, 'w') as f:
            # Write header
            f.writelines(header)
            # Write filtered lines
            f.writelines(filtered_lines)
        # Make the file executable
        os.chmod(output_file, 0o755)
        return True
    return False


def process_directory(base_dir, target_str: str, line_nums: list = None):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.sh'):
                input_path = os.path.join(root, file)
                suffix = f'_lines_{"-".join(map(str,line_nums))}' if line_nums else f'_{target_str}'
                output_path = os.path.join(root, file.replace('.sh', f'{suffix}.sh'))
                if filter_lines(input_path, output_path, target_str, line_nums):
                    print(f"Created file: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_str', type=str)
    parser.add_argument('--line_nums', type=int, nargs='+', help='Line numbers to select (1-based indexing)')
    args = parser.parse_args()
    
    # Use the same eval_config_dir from your original script
    eval_config_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/slurms/eval_slurms/"
    pipeline_config_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/slurms/pipe_slurms/"
    process_directory(eval_config_dir, args.target_str, args.line_nums)
    process_directory(pipeline_config_dir, args.target_str, args.line_nums)
