def to_string(l: List[str]) -> str:
<<<<<<< ./left.py
    if l is null or len(l) == 0:
        return ""
    return "__".join(l)
||||||| ./base.py
    if len(l) == 0:
        return ""
    return "..".join(l)
=======
    if len(l) == 0:
        return self.D
    return "__".join(l)
>>>>>>> ./right.py
