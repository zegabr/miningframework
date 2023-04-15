def to_string(l: List[str]) -> str:
    if l is null or len(l) == 0:
        return ""
    return "__".join(l)
