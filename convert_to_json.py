import json
import re
from pathlib import Path
from typing import Dict, Tuple

from src.utils.schem_utils import _initFromFile


def simplify_block_name(name: str) -> str:
    """Убирает атрибуты блока: 'minecraft:oak_stairs[...]' -> 'minecraft:oak_stairs'"""
    return re.sub(r"\[.*\]", "", name).strip()


def convert_to_rle_json(
    coord2byte: Dict[Tuple[int, int, int], int],
    palette: Dict[int, str],
    output_path: str,
    ignore_attributes: bool = False,
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

    # Чистим palette от нечисловых ключей (если остались)
    palette = {k: v for k, v in palette.items() if isinstance(k, int)}

    # 2. Если нужно, игнорируем атрибуты блоков
    if ignore_attributes:
        # Строим упрощённые имена
        simplified = {idx: simplify_block_name(name) for idx, name in palette.items()}
        # Уникальные упрощённые имена
        unique_names = sorted(set(simplified.values()))
        # Обеспечиваем, чтобы воздух был на индексе 0
        if "air" in unique_names or "minecraft:air" in unique_names:
            air_name = "minecraft:air" if "minecraft:air" in unique_names else "air"
            unique_names.remove(air_name)
            unique_names.insert(0, air_name)
        else:
            # Добавляем воздух, если его нет
            unique_names.insert(0, "minecraft:air")
        # Новая палитра: индекс -> упрощённое имя
        new_palette = {i: name for i, name in enumerate(unique_names)}
        # Маппинг старый индекс -> новый индекс
        name_to_new_idx = {name: i for i, name in enumerate(unique_names)}
        old_to_new = {idx: name_to_new_idx[simplified[idx]] for idx in palette}
        # Обновляем coord2byte
        new_coord2byte = {}
        for coord, idx in coord2byte.items():
            new_idx = old_to_new.get(idx)
            if new_idx is not None:
                new_coord2byte[coord] = new_idx
            # Если вдруг idx нет в старом palette, оставляем 0 (воздух)
            else:
                new_coord2byte[coord] = 0
        coord2byte = new_coord2byte
        palette = new_palette

    # 3. Нормализация: воздух должен быть индексом 0 (на случай, если выше не сделали)
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

    # 4. Построение слоёв RLE
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


def convert_to_2d_layers_json(
    coord2byte: Dict[Tuple[int, int, int], int],
    palette: Dict[int, str],
    output_path: str,
    ignore_attributes: bool = False,
) -> None:
    """Сохраняет JSON с палитрой, размерами и 2D-слоями индексов блоков."""

    # 1. Определяем границы
    if not coord2byte:
        raise ValueError("coord2byte не может быть пустым")
    xs, ys, zs = zip(*coord2byte.keys())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    depth = max_z - min_z + 1

    # Оставляем только числовые ключи палитры
    palette = {k: v for k, v in palette.items() if isinstance(k, int)}

    # 2. Игнорирование атрибутов (если нужно)
    if ignore_attributes:
        simplified = {idx: simplify_block_name(name) for idx, name in palette.items()}
        unique_names = sorted(set(simplified.values()))
        # Воздух на первое место
        air_candidates = [n for n in unique_names if n in ("air", "minecraft:air")]
        for air in air_candidates:
            unique_names.remove(air)
        unique_names.insert(0, "minecraft:air" if air_candidates else "minecraft:air")
        new_palette = {i: name for i, name in enumerate(unique_names)}
        name_to_new_idx = {name: i for i, name in enumerate(unique_names)}
        old_to_new = {idx: name_to_new_idx[simplified[idx]] for idx in palette}
        new_coord2byte = {}
        for coord, idx in coord2byte.items():
            new_coord2byte[coord] = old_to_new.get(idx, 0)
        coord2byte = new_coord2byte
        palette = new_palette

    # 3. Гарантируем, что воздух имеет индекс 0
    if 0 not in palette or palette[0] not in ("air", "minecraft:air"):
        air_idx = None
        for idx, name in palette.items():
            if name in ("air", "minecraft:air"):
                air_idx = idx
                break
        if air_idx is not None:
            old_name = palette[air_idx]
            name_at_0 = palette.get(0)
            palette[0] = old_name
            palette[air_idx] = name_at_0 if name_at_0 is not None else old_name
            remap = {}
            for coord, idx in coord2byte.items():
                if idx == air_idx:
                    remap[coord] = 0
                elif idx == 0:
                    remap[coord] = air_idx
                else:
                    remap[coord] = idx
            coord2byte = remap
        else:
            # воздуха нет вообще – добавляем
            new_palette = {0: "minecraft:air"}
            shift = {idx: idx + 1 for idx in palette}
            for old_idx, new_idx in shift.items():
                new_palette[new_idx] = palette[old_idx]
            palette = new_palette
            coord2byte = {c: shift[idx] for c, idx in coord2byte.items()}

    palette_dict = {str(k): v for k, v in palette.items()}

    # 4. Строим послойные 2D-массивы индексов
    layers_2d = []
    for y in range(min_y, max_y + 1):
        layer = []
        for z in range(min_z, max_z + 1):
            row = [coord2byte.get((x, y, z), 0) for x in range(min_x, max_x + 1)]
            layer.append(row)
        layers_2d.append(layer)

    # --- Специальная сериализация: каждый Z-ряд в одну строку ---
    # Создаём структуру с плейсхолдерами для рядов
    placeholder_layers = []
    for y, layer in enumerate(layers_2d):
        placeholder_layer = []
        for z, row in enumerate(layer):
            placeholder_layer.append(f"__ROW_{y}_{z}__")
        placeholder_layers.append(placeholder_layer)

    placeholder_result = {
        "palette": palette_dict,
        "size": {"width": width, "depth": depth, "height": height},
        "layers": placeholder_layers,
    }

    # Стандартная сериализация с отступами (ряды пока в виде строк)
    json_str = json.dumps(placeholder_result, ensure_ascii=False, indent=2)

    # Заменяем каждый плейсхолдер на компактный JSON ряда
    for y, layer in enumerate(layers_2d):
        for z, row in enumerate(layer):
            row_json = json.dumps(row, ensure_ascii=False)  # [0, 1, 2, 3]
            placeholder = f'"__ROW_{y}_{z}__"'
            json_str = json_str.replace(placeholder, row_json)

    # Запись результата
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)
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
    parser.add_argument(
        "--no-attrs",
        "--ignore-attributes",
        action="store_true",
        dest="ignore_attrs",
        help="Игнорировать атрибуты блоков, объединяя все состояния одного блока в один",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="rle",
        help="Режим работы скрипта: 'rle' слои или '2d' слои",
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

    if args.mode == "rle":
        convert_to_rle_json(
            coord2byte, palette, output_path, ignore_attributes=args.ignore_attrs
        )
    elif args.mode == "2d":
        convert_to_2d_layers_json(
            coord2byte, palette, output_path, ignore_attributes=args.ignore_attrs
        )
    else:
        print(f"No such mode: {args.mode}")
