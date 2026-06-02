import os
from pathlib import Path

import torch
from torch.utils.data import Sampler
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.utils.model_utils import get_head_key


class VAEDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self,
        name="train",
        block_data_path="src/block_data",
        mode="synth",
        web_factor=1,
        *args,
        **kwargs,
    ):
        """
        Args:
            name (str): partition name
            block_data_path (str): path to directory with block data
            mode (str): 'synth', 'web' or 'mix' - the mode of items sampling
            web_factor (int): the
        """
        self.index_path = ROOT_PATH / "data" / "dataset" / f"vae_{name}_index.json"
        self.web_index_path = (
            ROOT_PATH / "data" / "dataset" / f"vae_web_{name}_index.json"
        )

        self.web_factor = web_factor

        self.non_default_attribute_pairs = read_json(
            ROOT_PATH / block_data_path / "non_default_attribute_pairs.json"
        )

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        web_index = []
        index = []
        if mode == "synth" or mode == "mix":
            if self.index_path.exists():
                index = read_json(str(self.index_path))
            else:
                index = self._create_index(name)
        if mode == "web" or mode == "mix":
            if self.web_index_path.exists():
                web_index = read_json(str(self.web_index_path))
            else:
                web_index = self._create_index(f"web_{name}", web=True)

        if mode == "synth":
            indexes = {"synth": index}
        elif mode == "mix":
            indexes = {"synth": index, "web": web_index}
        elif mode == "web":
            indexes = {"web": web_index}
        else:
            raise RuntimeError(f"No such sampling mode: {mode}")

        super().__init__(indexes, *args, **kwargs)

    def _create_index(self, name, web=False):
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
        data_path: Path = ROOT_PATH / "data" / "dataset" / name
        data_path.mkdir(exist_ok=True, parents=True)

        # In this example, we create a synthesized dataset. However, in real
        # tasks, you should process dataset metadata and append it
        # to index. See other branches.
        for structure in tqdm(
            os.listdir(data_path), desc=f"Creating Vae Dataset: {name}"
        ):
            structire_path = data_path / structure

            if not (structire_path.is_dir()):
                continue

            if ("attributes_data.pt" not in os.listdir(structire_path)) or (
                "block_type.pt" not in os.listdir(structire_path)
            ):
                continue

            # parse dataset metadata and append it to index
            index.append(
                {
                    "structire_path": str(structire_path),
                }
            )

        # write index to disk
        if web:
            write_json(index, self.web_index_path)
        else:
            write_json(index, self.index_path)

        return index


class VAESampler(Sampler):
    def __init__(self, dataset):
        self.web_len = len(dataset._web_index)
        self.synth_len = len(dataset._index)

        self.web_factor = dataset.web_factor

    def __iter__(self):
        total = self.synth_len + self.web_len
        if self.web_len == 0 or self.synth_len == 0:
            return iter(torch.randperm(total).tolist())
        p_synth = 1 / (self.web_factor + 0.3 + 1)  # 0.3 is hardcoded

        n_synth_samples = int(total * p_synth)
        n_web_samples = total - n_synth_samples

        synth_idx = torch.randint(0, self.synth_len, (n_synth_samples,))
        web_idx = torch.randint(
            self.synth_len, self.synth_len + self.web_len, (n_web_samples,)
        )
        indices = torch.cat([synth_idx, web_idx])
        indices = indices[torch.randperm(len(indices))]
        return iter(indices.tolist())

    def __len__(self):
        return self.synth_len + self.web_len
