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
MAX_AIR_THRESHOLD = 0.9
MIN_Y_COORD = -57
MAX_Y_COORD = 316
CONV_WIDTH = 5
LOW_IDX_OFFSET = 2

NUM_PROBES = 16

NUM_TRIES = 10

LEVEL_DIMENSION = "minecraft:overworld"
AIR_BLOCK = "universal_minecraft:air"


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
    # 1. Семплируем границы по X и Z
    x1 = randint(minx, maxx - MIN_SELECTION_DIM)
    x2 = x1 + randint(MIN_SELECTION_DIM, min(MAX_SELECTION_DIM, maxx - x1))

    z1 = randint(minz, maxz - MIN_SELECTION_DIM)
    z2 = z1 + randint(MIN_SELECTION_DIM, min(MAX_SELECTION_DIM, maxz - z1))

    # 2. Находим максимальную высоту поверхности, сканируя несколько столбцов сверху вниз
    max_surface_y = MIN_Y_COORD

    for _ in range(NUM_PROBES):
        sx = randint(x1, x2 - 1)
        sz = randint(z1, z2 - 1)
        for y in range(MAX_Y_COORD - 1, MIN_Y_COORD - 1, -1):
            if str(level.get_block(sx, y, sz, dimension=LEVEL_DIMENSION)) != AIR_BLOCK:
                max_surface_y = max(max_surface_y, y + 1)
                break

    # 3. Небольшой запас воздуха над поверхностью (чтобы была видна граница земля/воздух)
    top_y = min(max_surface_y + MAX_SELECTION_DIM, MAX_Y_COORD)

    # если слишком низко верхний блок по y
    if top_y - MIN_Y_COORD < MIN_SELECTION_DIM:
        return None

    for _ in range(NUM_TRIES):
        # 4. Семплируем Y: y2 привязан к поверхности, y1 — ниже
        y2 = randint(min(MIN_Y_COORD + MIN_SELECTION_DIM, top_y), top_y)
        y1_min = max(MIN_Y_COORD, y2 - MAX_SELECTION_DIM)
        y1_max = y2 - MIN_SELECTION_DIM
        if y1_min > y1_max:
            continue
        y1 = randint(y1_min, y1_max)

        # 5. Проверяем долю воздуха
        volume = (x2 - x1) * (y2 - y1) * (z2 - z1)
        air_count = 0
        for x in range(x1, x2):
            for y in range(y1, y2):
                for z in range(z1, z2):
                    if (
                        str(level.get_block(x, y, z, dimension=LEVEL_DIMENSION))
                        == AIR_BLOCK
                    ):
                        air_count += 1

        air_ratio = air_count / volume
        if MIN_AIR_THRESHOLD <= air_ratio <= MAX_AIR_THRESHOLD:
            break
    else:
        return None

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
