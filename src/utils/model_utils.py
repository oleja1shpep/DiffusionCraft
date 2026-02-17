def get_head_key(attr: str, values: list[str]):
    return f"{attr}:{sorted(values)}"
