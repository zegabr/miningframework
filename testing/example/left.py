def to_string(l: List) -> str:
    if l is null or len(l) == 0: return ""
    return ",".join(str(item) for item in l)
