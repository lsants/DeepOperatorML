def multi_eval(v:tuple) -> tuple:
    f = lambda x: eval(x) if type(x) == str else x
    return tuple(map(f, v))