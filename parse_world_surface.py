import argparse
import os
from bisect import bisect_left
from pathlib import Path
from random import randint

from amulet import SelectionBox, SelectionGroup, load_level
from amulet.level.formats.sponge_schem import SpongeSchemFormatWrapper
from tqdm import tqdm

MIN_SELECTION_DIM = 16
MAX_SELECTION_DIM = 16 * 4

MIN_AIR_THRESHOLD = 0.1
MIN_Y_COORD = 20
MAX_Y_COORD = 201
CONV_WIDTH = 5
LOW_IDX_OFFSET = 2

LEVEL_DIMENSION = "minecraft:overworld"


def create_parser():
    parser = argparse.ArgumentParser(description="Generate world dataset")

    parser.add_argument(
        "--world-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--rx",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--rz",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--n-samples",
        required=True,
        type=int,
    )
    return parser


def generate_dimensions(level, minx, maxx, minz, maxz):
    def get_air_blocks_p_in_selection(x1, x2, y1, y2, z1, z2):
        AIR_BLOCK = "universal_minecraft:air"
        volume = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
        air_count = 0
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                for z in range(z1, z2 + 1):
                    if (
                        str(level.get_block(x, y, z, dimension=LEVEL_DIMENSION))
                        == AIR_BLOCK
                    ):
                        air_count += 1
        return air_count / volume

    def find_leftmost_air_idx(air_blocks):
        idx = bisect_left([t[1] for t in air_blocks], 1.0)
        if idx < len(air_blocks) and air_blocks[idx][1] == 1.0:
            return idx
        return None

    x1 = randint(minx, maxx - MAX_SELECTION_DIM)
    x2 = x1 + randint(MIN_SELECTION_DIM, MAX_SELECTION_DIM)
    z1 = randint(minz, maxz - MAX_SELECTION_DIM)
    z2 = z1 + randint(MIN_SELECTION_DIM, MAX_SELECTION_DIM)
    air_blocks = []
    for y_low, y_high in zip(
        range(MIN_Y_COORD, MAX_Y_COORD - CONV_WIDTH),
        range(MIN_Y_COORD + CONV_WIDTH, MAX_Y_COORD),
    ):
        air_blocks_moving_avg = get_air_blocks_p_in_selection(
            x1, x2, y_low, y_high, z1, z2
        )
        air_blocks.append((y_low, air_blocks_moving_avg))
    lowest_air_idx = find_leftmost_air_idx(air_blocks)
    if not lowest_air_idx:
        return None
    y2 = air_blocks[lowest_air_idx][0] + CONV_WIDTH
    y1 = air_blocks[lowest_air_idx][0]

    while lowest_air_idx > 0 and air_blocks[lowest_air_idx][1] > MIN_AIR_THRESHOLD:
        lowest_air_idx -= 1
    lowest_air_idx = max(lowest_air_idx - LOW_IDX_OFFSET, 0)
    y1 = air_blocks[lowest_air_idx][0]

    # ограничение на высоту
    height = y2 - y1
    if height < MIN_SELECTION_DIM:
        y1 = max(y2 - MIN_SELECTION_DIM, MIN_Y_COORD)
        if y2 - y1 < MIN_SELECTION_DIM:
            return None
    elif height > MAX_SELECTION_DIM:
        y1 = y2 - MAX_SELECTION_DIM

    return x1, x2, y1, y2, z1, z2


def save_selection(level, selection, path):
    wrapper = SpongeSchemFormatWrapper(path)
    wrapper.create_and_open(
        platform="java", version=(1, 20, 1), bounds=selection, overwrite=True
    )
    wrapper.translation_manager = level.translation_manager
    wrapper_dimension = wrapper.dimensions[0]
    for cx, cz in selection.chunk_locations():
        try:
            chunk = level.get_chunk(cx, cz, LEVEL_DIMENSION)
            wrapper.commit_chunk(chunk, wrapper_dimension)
        except Exception:
            continue
    wrapper.save()
    wrapper.close()


def create_sample(level, world_name, minx, maxx, minz, maxz, output_dir_path, index):
    dimensions = generate_dimensions(level, minx, maxx, minz, maxz)
    if not dimensions:
        return
    x1, x2, y1, y2, z1, z2 = dimensions
    output_path = os.path.join(
        output_dir_path, f"{world_name}_{x1}_{x2}_{y1}_{y2}_{z1}_{z2}.schem"
    )
    save_selection(
        level, SelectionGroup(SelectionBox((x1, y1, z1), (x2, y2, z2))), output_path
    )


def main(args):
    minx, maxx = -args.rx, args.rx
    minz, maxz = -args.rz, args.rz

    level = load_level(args.world_dir)
    world_name = Path(args.world_dir).stem
    output_dir_path = os.path.join(".", args.output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    for i in tqdm(range(args.n_samples)):
        create_sample(level, world_name, minx, maxx, minz, maxz, output_dir_path, i)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
