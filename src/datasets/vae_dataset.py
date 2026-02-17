import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class VAEDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(self, name="train", *args, **kwargs):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        self.index_path = ROOT_PATH / "data" / "dataset" / name / "vae_index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if self.index_path.exists():
            index = read_json(str(self.index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "dataset" / name
        data_path.mkdir(exist_ok=True, parents=True)

        block_type_dir = data_path / "block_type_tensors"
        attributes_dir = data_path / "attributes_data"

        # In this example, we create a synthesized dataset. However, in real
        # tasks, you should process dataset metadata and append it
        # to index. See other branches.
        for file in tqdm(os.listdir(block_type_dir), desc="Creating Vae Dataset"):
            block_type_path = block_type_dir / file
            attributes_path = attributes_dir / f"{Path(file).stem}.json"

            # parse dataset metadata and append it to index
            index.append(
                {
                    "block_type_path": str(block_type_path),
                    "attributes_path": str(attributes_path),
                }
            )

        # write index to disk
        write_json(index, self.index_path)

        return index
