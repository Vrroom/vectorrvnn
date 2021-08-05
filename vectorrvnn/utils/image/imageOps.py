from PIL import Image
import numpy as np

def imsave (arr, fname) : 
    """ utility for saving numpy array """ 
    if arr.dtype in [np.float32, np.float64] : 
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype == np.int: 
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    Image.fromarray(arr, chanType).save(fname)
