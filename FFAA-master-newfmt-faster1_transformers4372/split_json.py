import json
import math
import os
import random

def split_json_80_20(input_path, shuffle=False, output_dir=None, prefix=None):
    # Read input JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Make sure input is a list
    if not isinstance(data, list):
        raise ValueError("This script assumes the JSON file contains a top-level list.")

    total = len(data)
    if total == 0:
        raise ValueError("Input JSON list is empty.")

    # Optionally shuffle before splitting
    if shuffle:
        random.shuffle(data)

    # Compute 80/20 split index
    split_index = math.floor(total * 0.8)

    part1 = data[:split_index]   # 80%
    part2 = data[split_index:]   # 20%

    # Output directory and filenames
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(input_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    path_80 = os.path.join(output_dir, f"eFFAA_train_dataset.json")
    path_20 = os.path.join(output_dir, f"mids_train.json")

    # Write 80% file
    with open(path_80, "w", encoding="utf-8") as f:
        json.dump(part1, f, ensure_ascii=False, indent=2)

    # Write 20% file
    with open(path_20, "w", encoding="utf-8") as f:
        json.dump(part2, f, ensure_ascii=False, indent=2)

    print(f"Saved 80% file: {path_80} ({len(part1)} items)")
    print(f"Saved 20% file: {path_20} ({len(part2)} items)")


if __name__ == "__main__":
    # Example usage — change input_path
    split_json_80_20("/datasets/newout/eFFAA_dataset.json", shuffle=True)
