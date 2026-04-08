import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH, read_json
from src.utils.model_utils import get_head_key, load_checkpoint
from src.utils.schem_utils import create_schem, parse_schem

warnings.filterwarnings("ignore", category=UserWarning)


def move_batch_to_device(batch, device_tensors, device):
    """
    Move all necessary tensors to the device.

    Args:
        batch (dict): dict-based batch containing the data from
            the dataloader.
    Returns:
        batch (dict): dict-based batch containing the data from
            the dataloader with some of the tensors on the device.
    """
    for tensor_for_device in device_tensors:
        if tensor_for_device in ["attributes_masks", "attributes_values"]:
            for key in batch[tensor_for_device]:
                batch[tensor_for_device][key] = batch[tensor_for_device][key].to(device)
        else:
            batch[tensor_for_device] = batch[tensor_for_device].to(device)
    return batch


@hydra.main(version_base=None, config_path="src/configs", config_name="reconstruct")
def main(config):
    """
    Main script for reconstruction of a list of schematics. Instantiates the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.seed)

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    model.post_init(device)
    if config.get("model_path", None) is not None:
        model = load_checkpoint(model, config.model_path, device)

    # save_path for model predictions
    input_dir = ROOT_PATH / config.input_dir
    input_dir.mkdir(exist_ok=True, parents=True)

    output_dir = ROOT_PATH / config.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    block_data_dir = ROOT_PATH / config.block_data_dir
    non_default_attribute_pairs = read_json(
        block_data_dir / "non_default_attribute_pairs.json"
    )

    for file in tqdm(os.listdir(input_dir), desc="Reconsctructing..."):
        shem_path = input_dir / file
        if shem_path.suffix == ".schem":
            # turn schematic into tensors
            block_type_grid, attributes_data = parse_schem(shem_path, block_data_dir)

            attributes_values = dict()
            attributes_masks = dict()

            for attr, values in non_default_attribute_pairs:
                head_key = get_head_key(attr, values)

                attributes_values[head_key] = attributes_data[head_key]["values"].to(
                    torch.long
                )
                attributes_masks[head_key] = attributes_data[head_key]["mask"]

            # create batch of 1 element
            batch = collate_fn(
                [
                    {
                        "block_type_grid": block_type_grid,
                        "attributes_values": attributes_values,
                        "attributes_masks": attributes_masks,
                        "idx": -1,
                    }
                ],
                config.model.num_layers,
            )

            batch = move_batch_to_device(batch, config.device_tensors, device)

            with torch.no_grad():
                outputs = model(**batch)

            pred_attributes_data = dict()

            for attr, values in non_default_attribute_pairs:
                head_key = get_head_key(attr, values)
                pred_attributes_data[head_key] = {}
                pred_attributes_data[head_key]["mask"] = (
                    outputs["pred_attribures_masks"][head_key][0].detach().cpu()
                )
                pred_attributes_data[head_key]["values"] = (
                    outputs["attributes_logits"][head_key].detach().argmax(-1).cpu()
                )

            create_schem(
                outputs["pred_block_type_grid"][0].detach().cpu(),
                pred_attributes_data,
                output_path=output_dir / file,
                block_data_dir=block_data_dir,
            )


if __name__ == "__main__":
    main()
