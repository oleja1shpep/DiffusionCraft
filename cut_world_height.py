import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.utils.model_utils import BLOCK_TYPE

HEIGHT_LIMIT = 64


def create_parser():
    parser = argparse.ArgumentParser(description="Cut World Height")

    parser.add_argument(
        "--data-dir",
        required=True,
    )

    return parser


def main(args):
    data_dir = Path(args.data_dir)
    for folder in tqdm(os.listdir(data_dir)):
        block_type_grid = torch.load(
            data_dir / folder / f"{BLOCK_TYPE}.pt", weights_only=False
        )

        height = block_type_grid.shape[1]
        if height <= HEIGHT_LIMIT:
            continue

        print(f"Processing: {folder}")

        block_type_grid = block_type_grid[:, -HEIGHT_LIMIT:]

        attributes_data = torch.load(
            data_dir / folder / "attributes_data.pt", weights_only=False
        )

        for head_key in attributes_data:
            mask: torch.Tensor = attributes_data[head_key]["mask"]
            values = attributes_data[head_key]["values"]
            idxs = mask.nonzero()

            values = values[idxs[:, 1] >= height - HEIGHT_LIMIT]
            mask = mask[:, -HEIGHT_LIMIT:]

            attributes_data[head_key]["mask"] = mask
            attributes_data[head_key]["values"] = values

        torch.save(block_type_grid, data_dir / folder / f"{BLOCK_TYPE}.pt")
        torch.save(attributes_data, data_dir / folder / "attributes_data.pt")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
