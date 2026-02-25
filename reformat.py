"""
The script parses .schem files into tensors. The output has the following structure

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
"""


import argparse
import io
import json
import os
import shutil
from pathlib import Path

import nbtlib
import numpy as np
import torch
from immutable_views import *
from nbtlib.tag import *
from tqdm import tqdm

from src.utils.model_utils import AIR, AIR_BLOCK_IDX, BLOCK_TYPE, INFESTED, get_head_key


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
                import pdb

                pdb.set_trace()
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
