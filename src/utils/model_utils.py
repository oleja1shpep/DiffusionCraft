from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from PIL import Image

INFESTED = "minecraft:infested_"
PALE_OAK = "pale_oak"
LIGHTNING_ROD = "lightning_rod"
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


def build_block_grid_views(
    block_type_grid: np.ndarray,
    block2color: dict[str, list[int]],
    idx2block: list[str],
    gradient=0,
) -> dict[str, np.ndarray]:
    """
    Build RGB arrays for 6 orthographic views of a block grid.

    Args:
        block_type_grid (Array): an array of block idxs of shape (W, H, L).
        block2color (Dict): a dict with RGB color for each block.
        idx2block (List): a list for mapping idx to block name.
        gradient (Int): regulates the shadow for further blocks.
    Returns:
        dict with keys: top, bottom, front, back, left, right.
    """
    width, height, length = block_type_grid.shape

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

    return {
        "top": top_view,
        "bottom": bottom_view,
        "front": front_view,
        "back": back_view,
        "left": left_view,
        "right": right_view,
    }


def _view_array_to_image(view: np.ndarray) -> Image.Image:
    return Image.fromarray(np.flipud(view.astype(np.uint8)))


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
        PIL Image with 2x3 grid of views.
    """
    views = build_block_grid_views(block_type_grid, block2color, idx2block, gradient)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    fig: Figure

    add_image_to_axis(ax[0][0], views["top"], "Top (-y)", "x", "z")
    add_image_to_axis(ax[1][0], views["bottom"], "Bottom (+y)", "x", "z")

    add_image_to_axis(ax[0][1], views["back"], "Back (-x)", "z", "y")
    add_image_to_axis(ax[1][1], views["front"], "Front (+x)", "z", "y")

    add_image_to_axis(ax[0][2], views["left"], "Left (-z)", "x", "y")
    add_image_to_axis(ax[1][2], views["right"], "Right (+z)", "x", "y")

    buffer = BytesIO()
    fig.savefig(buffer, format="jpg", dpi=100)
    buffer.seek(0)
    plt.close(fig)

    return Image.open(buffer)


def save_block_grid_renders(
    block_type_grid: np.ndarray,
    block2color: dict[str, list[int]],
    idx2block: list[str],
    output_dir: Path,
    gradient=0,
) -> None:
    """
    Save combined and per-view JPG renders of a block grid.

    Args:
        block_type_grid (Array): an array of block idxs of shape (W, H, L).
        block2color (Dict): a dict with RGB color for each block.
        idx2block (List): a list for mapping idx to block name.
        output_dir (Path): directory to save render.jpg and individual views.
        gradient (Int): regulates the shadow for further blocks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    views = build_block_grid_views(block_type_grid, block2color, idx2block, gradient)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    add_image_to_axis(ax[0][0], views["top"], "Top (-y)", "x", "z")
    add_image_to_axis(ax[1][0], views["bottom"], "Bottom (+y)", "x", "z")
    add_image_to_axis(ax[0][1], views["back"], "Back (-x)", "z", "y")
    add_image_to_axis(ax[1][1], views["front"], "Front (+x)", "z", "y")
    add_image_to_axis(ax[0][2], views["left"], "Left (-z)", "x", "y")
    add_image_to_axis(ax[1][2], views["right"], "Right (+z)", "x", "y")

    buffer = BytesIO()
    fig.savefig(buffer, format="jpg", dpi=100)
    buffer.seek(0)
    plt.close(fig)
    Image.open(buffer).save(output_dir / "render.jpg")

    for name, view in views.items():
        _view_array_to_image(view).save(output_dir / f"{name}.jpg")


def make_class_weights(values: torch.Tensor, power=0.3, eps=1e-5) -> torch.Tensor:
    values = values.float()
    max_count = values.max()

    weights = (max_count / (values + eps)) ** power
    weights = weights / weights.max() * 0.9  # редчайшие классы -> 1
    return weights + 0.01  # Зажимаем между 0.01 и 1
