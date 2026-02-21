import torch

from src.utils.model_utils import AIR_BLOCK_IDX


def collate_fn(dataset_items: list[dict], n_layers=3) -> dict:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    max_width, max_height, max_length = 0, 0, 0
    result_batch["attributes_masks"] = dict()
    result_batch["attributes_values"] = dict()
    result_batch["block_type_grid"] = []

    for item in dataset_items:
        for key in item["attributes_masks"]:
            if key not in result_batch["attributes_masks"]:
                result_batch["attributes_masks"][key] = []
            result_batch["attributes_masks"][key].append(item["attributes_masks"][key])

            if key not in result_batch["attributes_values"]:
                result_batch["attributes_values"][key] = []
            result_batch["attributes_values"][key].append(
                item["attributes_values"][key]
            )

        width, height, length = item["block_type_grid"].shape
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        max_length = max(max_length, length)

        result_batch["block_type_grid"].append(item["block_type_grid"])

    scale_factor = 2**n_layers
    max_width += (scale_factor - max_width % scale_factor) % scale_factor
    max_height += (scale_factor - max_height % scale_factor) % scale_factor
    max_length += (scale_factor - max_length % scale_factor) % scale_factor

    for i in range(len(dataset_items)):
        block_grid = result_batch["block_type_grid"][i]
        width, height, length = block_grid.shape

        # padding
        block_grid = torch.concatenate(
            [
                block_grid,
                AIR_BLOCK_IDX
                * torch.ones(max_width - width, height, length, dtype=torch.long),
            ],
            dim=0,
        )
        # do not forget to pad masks with False (since air doesn't have attributes)
        for key in result_batch["attributes_masks"]:
            result_batch["attributes_masks"][key][i] = torch.concatenate(
                [
                    result_batch["attributes_masks"][key][i],
                    torch.zeros(max_width - width, height, length, dtype=torch.bool),
                ],
                dim=0,
            )

        block_grid = torch.concatenate(
            [
                block_grid,
                AIR_BLOCK_IDX
                * torch.ones(max_width, max_height - height, length, dtype=torch.long),
            ],
            dim=1,
        )
        for key in result_batch["attributes_masks"]:
            result_batch["attributes_masks"][key][i] = torch.concatenate(
                [
                    result_batch["attributes_masks"][key][i],
                    torch.zeros(
                        max_width, max_height - height, length, dtype=torch.bool
                    ),
                ],
                dim=1,
            )

        block_grid = torch.concatenate(
            [
                block_grid,
                AIR_BLOCK_IDX
                * torch.ones(
                    max_width, max_height, max_length - length, dtype=torch.long
                ),
            ],
            dim=2,
        ).unsqueeze(
            0
        )  # for vstack
        for key in result_batch["attributes_masks"]:
            result_batch["attributes_masks"][key][i] = torch.concatenate(
                [
                    result_batch["attributes_masks"][key][i],
                    torch.zeros(
                        max_width, max_height, max_length - length, dtype=torch.bool
                    ),
                ],
                dim=2,
            ).unsqueeze(0)

        result_batch["block_type_grid"][i] = block_grid

    result_batch["block_type_grid"] = torch.vstack(result_batch["block_type_grid"])
    for key in result_batch["attributes_masks"]:
        result_batch["attributes_masks"][key] = torch.vstack(
            result_batch["attributes_masks"][key]
        )
        result_batch["attributes_values"][key] = torch.concatenate(
            result_batch["attributes_values"][key]
        )  # just 1D tensor

    return result_batch
