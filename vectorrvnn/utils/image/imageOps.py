from PIL import Image
import numpy as np
import base64, io

ENCODING = 'utf-8'

def imsave (arr, fname) : 
    """ utility for saving numpy array """ 
    imgToPIL(arr).save(fname)

def imgArrayToPIL (arr) :
    """ utility to convert img array to PIL """
    if arr.dtype in [np.float32, np.float64, float] :
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype in [np.int32, np.int64, int]:
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType)

def aspectRatioPreservingResize (arr, smaller_dim) :
    """ utility for resizing image, ensuring that smaller dimension matches """
    pil_img = imgArrayToPIL(arr)
    h, w = pil_img.size
    if h < w :
        h, w = smaller_dim, smaller_dim * w / h
    else :
        h, w = smaller_dim * h / w, smaller_dim
    h, w = int(h), int(w)
    resized = pil_img.resize((h, w))
    np_arr = np.array(resized).astype(arr.dtype)
    if arr.dtype in [float, np.float32, np.float64] :
        np_arr /= 255.0
    return np_arr
    
def imgArrayToPILRGB (arr) : 
    """ utility to convert img array to 3 channel PIL """
    from vectorrvnn.utils import alphaComposite
    arr = alphaComposite(arr)
    if arr.dtype in [np.float32, np.float64] : 
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype == np.int: 
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType).convert("RGB")

def bytes2PIL(b64Str):
    imgdata = base64.b64decode(str(b64Str))
    return Image.open(io.BytesIO(imgdata))

def PIL2byteStr(img):
    img = np.array(img)
    buffered = io.BytesIO()
    image = Image.fromarray(img)
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode(ENCODING)
