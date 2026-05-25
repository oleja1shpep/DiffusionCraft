import json
from pathlib import Path
from typing import Dict, Tuple

import nbtlib

from src.utils.schem_utils import _initFromFile


def convert_to_rle_json(
    coord2byte: Dict[Tuple[int, int, int], int],
    palette: Dict[int, str],
    output_path: str,
) -> None:
    # 1. Определяем границы
    if not coord2byte:
        raise ValueError("coord2byte не может быть пустым")
    xs, ys, zs = zip(*coord2byte.keys())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    length = max_z - min_z + 1

    # 2. Исправляем palette: только числовые ключи, воздух на индексе 0
    palette = {k: v for k, v in palette.items() if isinstance(k, int)}
    if 0 not in palette or palette[0] not in ("air", "minecraft:air"):
        air_idx = None
        for idx, name in palette.items():
            if name in ("air", "minecraft:air"):
                air_idx = idx
                break
        if air_idx is not None:
            old_air_name = palette[air_idx]
            block_at_0 = palette.get(0)
            palette[0] = old_air_name
            palette[air_idx] = block_at_0 if block_at_0 is not None else old_air_name
            new_coord2byte = {}
            for coord, idx in coord2byte.items():
                if idx == air_idx:
                    new_coord2byte[coord] = 0
                elif idx == 0:
                    new_coord2byte[coord] = air_idx
                else:
                    new_coord2byte[coord] = idx
            coord2byte = new_coord2byte
        else:
            new_palette = {0: "minecraft:air"}
            index_shift = {}
            for idx, name in palette.items():
                new_idx = idx + 1
                new_palette[new_idx] = name
                index_shift[idx] = new_idx
            palette = new_palette
            new_coord2byte = {}
            for coord, idx in coord2byte.items():
                new_coord2byte[coord] = index_shift[idx]
            coord2byte = new_coord2byte

    palette_dict = {str(k): v for k, v in palette.items()}

    # 3. Построение слоёв (без изменений)
    rle_layers = []
    for y in range(min_y, max_y + 1):
        layer_flat = []
        for z in range(min_z, max_z + 1):
            for x in range(min_x, max_x + 1):
                idx = coord2byte.get((x, y, z), 0)
                layer_flat.append(idx)
        parts = []
        if layer_flat:
            current = layer_flat[0]
            count = 1
            for val in layer_flat[1:]:
                if val == current:
                    count += 1
                else:
                    parts.append(f"{count}:{current}")
                    current = val
                    count = 1
            parts.append(f"{count}:{current}")
        rle_layers.append(",".join(parts))

    result = {
        "palette": palette_dict,
        "size": {"width": width, "height": height, "length": length},
        "rle_layers": rle_layers,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Файл сохранён: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Преобразование схематики Minecraft в JSON с послойным RLE-кодированием"
    )
    parser.add_argument("input", help="Путь к файлу схематики (.schem)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Путь к выходному JSON-файлу (по умолчанию: имя_входного.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: файл '{args.input}' не найден")
        exit(1)

    output_path = args.output if args.output else input_path.stem + ".json"

    try:
        coord2byte, palette = _initFromFile(str(input_path))
    except Exception as e:
        print(f"Ошибка при чтении схематики: {e}")
        exit(1)

    convert_to_rle_json(coord2byte, palette, output_path)
