import os
import re

def filter_nf4_lines(input_file, output_file):
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

    # Filter for lines containing --nf4_model
    nf4_lines = [line for line in content if '--nf4_model' in line]

    # If we found any NF4 lines, write the new file
    if nf4_lines:
        with open(output_file, 'w') as f:
            # Write header
            f.writelines(header)
            # Write NF4 lines
            f.writelines(nf4_lines)
        # Make the file executable
        os.chmod(output_file, 0o755)
        return True
    return False

def process_directory(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.sh'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, file.replace('.sh', '_nf4.sh'))
                if filter_nf4_lines(input_path, output_path):
                    print(f"Created NF4 file: {output_path}")

if __name__ == "__main__":
    # Use the same eval_config_dir from your original script
    eval_config_dir = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/config/eval_config/"
    process_directory(eval_config_dir)