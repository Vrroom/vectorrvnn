import numpy as np

def complexLE (x, y) :
    """
    Lesser than equal to for complex numbers.

    Parameters
    ----------
    x : complex
    y : complex
    """
    if x.real < y.real :
        return True
    elif x.real > y.real :
        return False
    else :
        return x.imag < y.imag

def complexDot (x, y) :
    """
    Dot product between complex numbers treating them as vectors in R^2.

    Parameters
    ----------
    x : complex
    y : complex
    """
    return (x.real * y.real) + (x.imag * y.imag)

def complexCross (x, y) : 
    """
    Cross product between complex numbers treating them as vectors in R^2.

    Parameters
    ----------
    x : complex
    y : complex
    """
    v1 = np.array([x.real, x.imag, 0.0]) 
    v2 = np.array([y.real, y.imag, 0.0]) 
    return np.linalg.norm(np.cross(v1,v2))

