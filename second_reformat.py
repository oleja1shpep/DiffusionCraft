"""
The script parses turns one file structure into another

FROM:

output_dir/
    structure_1/
        block_type.pt
        <attribute_pair_1>_values.pt
        <attribute_pair_1>_mask.pt

        <attribute_pair_2>_values.pt
        <attribute_pair_2>_mask.pt

        ...

        <attribute_pair_N>_values.pt
        <attribute_pair_N>_mask.pt
    ...

TO:

output_dir/
    structure_1/
        block_type.pt
        attributes_data.pt
    ...

"""


import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from src.utils.model_utils import get_head_key


def create_parser():
    parser = argparse.ArgumentParser(description="Generate world dataset")

    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--block-data-dir", required=True, type=str)

    return parser


def main(args):
    data_dir = Path(args.data_dir)

    with open(Path(args.block_data_dir) / "non_default_attribute_pairs.json") as f:
        non_default_attribute_pairs = json.load(f)

    for name in tqdm(os.listdir(data_dir)):
        attributes_data = dict()
        if (data_dir / name / "attributes_data.pt").exists():
            continue
        for attr, values in non_default_attribute_pairs:
            head_key = get_head_key(attr, values)

            try:
                mask = torch.load(
                    data_dir / name / f"{head_key}_mask.pt", weights_only=False
                )
                values = torch.load(
                    data_dir / name / f"{head_key}_values.pt", weights_only=False
                )
            except Exception as e:
                print(e)
                shutil.rmtree(data_dir / name)
                break

            attributes_data[head_key] = dict()
            attributes_data[head_key]["mask"] = mask  # bool
            attributes_data[head_key]["values"] = values  # int8

            os.remove(data_dir / name / f"{head_key}_mask.pt")
            os.remove(data_dir / name / f"{head_key}_values.pt")
        else:
            torch.save(attributes_data, data_dir / name / "attributes_data.pt")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
