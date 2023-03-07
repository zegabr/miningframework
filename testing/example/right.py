def to_string(l: List) -> str:
    if len(l) == 0: return D
    return ",".join(str(item) for item in l)
