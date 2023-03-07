def to_string(l: List) -> str:
<<<<<<< left.py
    if l is null or len(l) == 0: return ""
||||||| base.py
    if len(l) == 0: return ""
=======
    if len(l) == 0: return D
>>>>>>> right.py
    return ",".join(str(item) for item in l)
