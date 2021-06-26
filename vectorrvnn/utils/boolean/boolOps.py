def complement(predicate) :
    return lambda x : not predicate(x)

def implies(a, b) :
    return not a or b

