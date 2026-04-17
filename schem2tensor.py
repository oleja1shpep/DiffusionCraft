"""
The script parses .schem files into tensors. The output has the following structure

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
from pathlib import Path

import nbtlib
import torch
from tqdm import tqdm

from src.utils.schem_utils import (
    AIR,
    BLOCK_TYPE,
    _initFromFile,
    block_to_idx,
    create_block2idx_mapping,
    filter_attribute_dict,
    get_head_key,
    parse_block,
)


def create_parser():
    parser = argparse.ArgumentParser(description="Generate world dataset")

    parser.add_argument("--schem-dir", required=True, type=str)

    parser.add_argument("--output-dir", required=False, type=str)

    parser.add_argument(
        "--block-data-dir", required=False, type=str, default="./src/block_data"
    )

    parser.add_argument(
        "--n-workers",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--limit",
        required=False,
        type=int,
    )
    return parser


def parse_schematics(
    data_dir, output_dir=None, block_data_dir="src/block_data", limit=None
):
    """
    A function for turning schematics in folder into tensors
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir / "parsed"
    output_dir = Path(output_dir)

    block_data_dir = Path(block_data_dir)

    with open(block_data_dir / "filtered_blocks.json") as f:
        filtered_blocks_dict = json.load(f)

    if not (
        os.path.exists(block_data_dir / "idx2block.json")
        and os.path.exists(block_data_dir / "block2idx.json")
    ):
        create_block2idx_mapping(block_data_dir)

    with open(block_data_dir / "block2idx.json") as f:
        block2idx = json.load(f)

    with open(block_data_dir / "attributes_defaults.json") as f:
        attributes_defaults = json.load(f)
    with open(block_data_dir / "block_attributes_defaults.json") as f:
        block_attributes_defaults = json.load(f)

    with open(block_data_dir / "block_attributes_defaults.json") as f:
        block_attributes_defaults = json.load(f)

    with open(block_data_dir / "non_default_attribute_pairs.json") as f:
        non_default_attribute_pairs = json.load(f)

    with open(block_data_dir / "attr_pair2idxs.json") as f:
        attr_pair2idxs = json.load(f)

    for key in attr_pair2idxs:
        attr_pair2idxs[key] = torch.tensor(attr_pair2idxs[key], dtype=torch.int16)

    files = sorted(os.listdir(data_dir))

    if limit is None:
        limit = len(files)

    for i, schm in enumerate(tqdm(files, total=limit)):
        if i >= limit:
            break
        file: Path = data_dir / schm

        if file.suffix == ".schem":
            structure_name = file.stem
            if (output_dir / structure_name).exists():
                continue
            try:
                schem = nbtlib.load(file)
            except Exception as e:
                print(f"Error: {e}\nFilename: {schm}")
                continue
            try:
                if "Schematic" in schem.keys():  # rarely this key appears
                    schem = schem["Schematic"]
                length, width, height = schem["Length"], schem["Width"], schem["Height"]
            except Exception as e:
                print(f"Error: {e}\nFilename: {schm}")
                continue

            del schem

            coord2byte, palette = _initFromFile(file)

            block_grid_tensor = torch.zeros(
                (width, height, length), dtype=torch.int16
            )  # x, y, z

            attributes = {}

            for x, y, z in coord2byte:
                block_byte = coord2byte[(x, y, z)]
                block = palette[block_byte]

                block, attr_dict = parse_block(block)  # str, dict
                block, block_idx = block_to_idx(block, block2idx)  # str, int
                if block == AIR:
                    attr_dict = {}

                attr_dict = filter_attribute_dict(
                    block=block,
                    attr_dict=attr_dict,
                    attributes_defaults=attributes_defaults,
                    block_attributes_defaults=block_attributes_defaults,
                    filtered_blocks_dict=filtered_blocks_dict,
                )

                block_grid_tensor[x][y][z] = block_idx
                if len(attr_dict):
                    attributes[(x, y, z)] = attr_dict

            os.makedirs(output_dir / structure_name, exist_ok=True)
            torch.save(
                block_grid_tensor, output_dir / structure_name / f"{BLOCK_TYPE}.pt"
            )  # int16

            # create masks and attr vectors for each attr-value pair
            attributes_data = dict()
            for attr, values in non_default_attribute_pairs:
                head_key = get_head_key(attr, values)
                mask = torch.isin(block_grid_tensor, attr_pair2idxs[head_key])
                idxs = torch.nonzero(mask)

                attribute_values = []
                for x, y, z in idxs:
                    xyz_key = (x.item(), y.item(), z.item())
                    if (
                        attr not in attributes[xyz_key]
                    ):  # if attribute we predict does not exist in this block make it default 0
                        attributes[xyz_key][attr] = 0
                    attribute_values.append(attributes[xyz_key][attr])

                attributes_data[head_key] = dict()
                attributes_data[head_key]["mask"] = mask  # bool
                attributes_data[head_key]["values"] = torch.tensor(
                    attribute_values, dtype=torch.int8
                )  # int8

            torch.save(
                attributes_data, output_dir / structure_name / "attributes_data.pt"
            )


def main(args):
    data_dir = args.schem_dir
    output_dir = args.output_dir
    block_data_dir = args.block_data_dir
    limit = args.limit
    parse_schematics(data_dir, output_dir, block_data_dir, limit)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
