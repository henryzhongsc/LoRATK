import sys
import argparse


def split_slurm_file(input_file, num_splits):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Find the header (lines before "python ")
    header = []
    content = []
    for line in lines:
        if line.startswith("python "):
            content.append(line)
        elif line.startswith("accelerate "):
            content.append(line)
        else:
            header.append(line)

    # Calculate the number of lines per split
    lines_per_split = len(content) // num_splits
    remainder = len(content) % num_splits

    # Split the content
    start = 0
    for i in range(num_splits):
        end = start + lines_per_split + (1 if i < remainder else 0)

        output_filename = f"{input_file.rsplit('.', 1)[0]}_{i + 1}.slurm"
        with open(output_filename, 'w') as f:
            f.writelines(header)
            f.writelines(content[start:end])

        start = end

    print(f"Split {input_file} into {num_splits} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a Slurm file into multiple files.")
    parser.add_argument("input_file", help="Path to the input Slurm file")
    parser.add_argument("num_splits", type=int, help="Number of files to split into")

    args = parser.parse_args()

    split_slurm_file(args.input_file, args.num_splits)