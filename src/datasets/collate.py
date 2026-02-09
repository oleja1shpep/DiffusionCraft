import torch

AIR_BLOCK_IDX = 0


def collate_fn(dataset_items: list[dict]):
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

    # example of collate_fn
    result_batch["attributes_data"] = [
        dataset_items[i]["attributes_data"] for i in range(len(dataset_items))
    ]
    result_batch["block_type_grid"]
    max_width, max_height, max_length = 0, 0, 0
    for i in range(len(dataset_items)):
        width, height, length = dataset_items[i]["block_type_grid"].shape
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        max_length = max(max_length, length)

    result_batch["block_type_grid"] = []

    for i in range(len(dataset_items)):
        block_grid = dataset_items[i]["block_type_grid"]
        width, height, length = block_grid.shape

        # width_pad
        block_grid = torch.concatenate(
            [block_grid, AIR_BLOCK_IDX * torch.ones(max_width - width, 1, 1)], dim=0
        )
        block_grid = torch.concatenate(
            [block_grid, AIR_BLOCK_IDX * torch.ones(1, max_height - height, 1)], dim=1
        )
        block_grid = torch.concatenate(
            [block_grid, AIR_BLOCK_IDX * torch.ones(1, 1, max_length - length)], dim=2
        )

        result_batch["block_type_grid"].append(block_grid)
    result_batch["block_type_grid"] = torch.vstack(result_batch["block_type_grid"])
    return result_batch
