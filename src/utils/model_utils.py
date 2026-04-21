from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from PIL import Image

INFESTED = "minecraft:infested_"
AIR = "minecraft:air"
WATER = "minecraft:water"
BLOCK_TYPE = "block_type"
AIR_BLOCK_IDX = 0

matplotlib.use("Agg")


def load_checkpoint(model: torch.nn.Module, path: str, device):
    checkpoint = torch.load(path, device, weights_only=False)

    if checkpoint.get("state_dict") is not None:
        checkpoint = checkpoint["state_dict"]

    model_state_dict = model.state_dict()

    for key, value in checkpoint.items():
        key: str
        if key.startswith("module."):
            key = key[key.index(".") + 1 :]
        model_state_dict[key] = value

    model.load_state_dict(model_state_dict)
    return model


def get_head_key(attr: str, values: list[str]):
    return f"{attr}_{sorted(values)}"


def add_image_to_axis(ax: Axes, img, title, x_label, y_label):
    ax.imshow(img, origin="lower")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def render_block_grid(
    block_type_grid: np.ndarray,
    block2color: dict[str, list[int]],
    idx2block: list[str],
    gradient=0,
) -> Image.Image:
    """
    Function for rendering block grid from 6 perspectives

    Args:
        block_type_grid (Array) : an array of block idxs of shape (W, H, L).
        block2color (Dict) : a dict with RGB color for each block.
        idx2block (List) : a list for mapping idx to block name.
        gradient (Int) : regulates the shadow for further blocks.
    Returns:
        None
    """

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    fig: Figure

    width, height, length = block_type_grid.shape

    # TOP and BOTTOM
    top_view = np.zeros((length, width, 3), dtype=np.int32)
    bottom_view = np.zeros((length, width, 3), dtype=np.int32)

    for x in range(width):
        for z in range(length):
            highest_block_idx = AIR_BLOCK_IDX
            for y in range(height - 1, -1, -1):
                idx = block_type_grid[x, y, z]
                if idx != AIR_BLOCK_IDX:
                    highest_block_idx = idx
                    break
            color = block2color[idx2block[highest_block_idx]]
            top_view[z, x] = np.int32(np.maximum(0, np.array(color) - y * gradient))

            lowest_block_idx = AIR_BLOCK_IDX
            for y in range(height):
                idx = block_type_grid[x, y, z]
                if idx != AIR_BLOCK_IDX:
                    lowest_block_idx = idx
                    break
            color = block2color[idx2block[lowest_block_idx]]
            bottom_view[z, x] = np.int32(np.maximum(0, np.array(color) - y * gradient))

    # FRONT and BACK
    front_view = np.zeros((height, length, 3), dtype=np.int32)
    back_view = np.zeros((height, length, 3), dtype=np.int32)

    for y in range(height):
        for z in range(length):
            back_block_idx = AIR_BLOCK_IDX
            for x in range(width - 1, -1, -1):
                idx = block_type_grid[x, y, z]
                if idx != AIR_BLOCK_IDX:
                    back_block_idx = idx
                    break
            color = block2color[idx2block[back_block_idx]]
            back_view[y, z] = np.int32(np.maximum(0, np.array(color) - y * gradient))

            front_block_idx = AIR_BLOCK_IDX
            for x in range(width):
                idx = block_type_grid[x, y, z]
                if idx != AIR_BLOCK_IDX:
                    front_block_idx = idx
                    break
            color = block2color[idx2block[front_block_idx]]
            front_view[y, z] = np.int32(np.maximum(0, np.array(color) - y * gradient))

    # RIGHT and LEFT
    right_view = np.zeros((height, width, 3), dtype=np.int32)
    left_view = np.zeros((height, width, 3), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            left_block_idx = AIR_BLOCK_IDX
            for z in range(length - 1, -1, -1):
                idx = block_type_grid[x, y, z]
                if idx != AIR_BLOCK_IDX:
                    left_block_idx = idx
                    break
            color = block2color[idx2block[left_block_idx]]
            left_view[y, x] = np.int32(np.maximum(0, np.array(color) - y * gradient))

            right_block_idx = AIR_BLOCK_IDX
            for z in range(length):
                idx = block_type_grid[x, y, z]
                if idx != AIR_BLOCK_IDX:
                    right_block_idx = idx
                    break

            color = block2color[idx2block[right_block_idx]]
            right_view[y, x] = np.int32(np.maximum(0, np.array(color) - y * gradient))

    add_image_to_axis(ax[0][0], top_view, "Top (-y)", "x", "z")
    add_image_to_axis(ax[1][0], bottom_view, "Bottom (+y)", "x", "z")

    add_image_to_axis(ax[0][1], back_view, "Back (-x)", "z", "y")
    add_image_to_axis(ax[1][1], front_view, "Front (+x)", "z", "y")

    add_image_to_axis(ax[0][2], left_view, "Left (-z)", "x", "y")
    add_image_to_axis(ax[1][2], right_view, "Right (+z)", "x", "y")

    buffer = BytesIO()
    fig.savefig(buffer, format="jpg", dpi=100)
    buffer.seek(0)
    plt.close(fig)

    # Step 3: Load the buffer into a PIL Image
    return Image.open(buffer)


def make_class_weights(
    values: torch.Tensor, power=0.3, eps=1e-5, scale=1
) -> torch.Tensor:
    values = values.float()
    max_count = values.max()

    weights = (max_count / (values + eps)) ** power
    weights = weights / weights.max() * 9.9  # редчайшие классы -> 10
    return (weights + 0.1) * scale
