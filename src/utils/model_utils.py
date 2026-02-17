INFESTED = "minecraft:infested_"
AIR = "minecraft:air"
BLOCK_TYPE = "block_type"
AIR_BLOCK_IDX = 0


def get_head_key(attr: str, values: list[str]):
    return f"{attr}_{sorted(values)}"
