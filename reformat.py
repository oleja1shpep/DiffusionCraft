"""
The script parses turns one file structure into another

FROM:

output_dir/
    structure_1/
        block_type.pt
        attribute_pair_1/
            values.pt
            mask.pt
        attribute_pair_2/
            values.pt
            mask.pt

        ...

        attribute_pair_N/
            values.pt
            mask.pt
    ...

TO:

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
"""


import argparse
import json
import os
import shutil
from pathlib import Path

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
        for attr, values in non_default_attribute_pairs:
            head_key = get_head_key(attr, values)
            if (data_dir / name / f"{head_key}_mask.pt").exists() and (
                data_dir / name / f"{head_key}_values.pt"
            ).exists():
                continue

            if not (
                (data_dir / name / head_key / "mask.pt").exists()
                and (data_dir / name / head_key / "values.pt").exists()
            ):
                shutil.rmtree(data_dir / name)
                break
            os.rename(
                data_dir / name / head_key / "mask.pt",
                data_dir / name / f"{head_key}_mask.pt",
            )

            os.rename(
                data_dir / name / head_key / "values.pt",
                data_dir / name / f"{head_key}_values.pt",
            )

            os.rmdir(data_dir / name / head_key)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
