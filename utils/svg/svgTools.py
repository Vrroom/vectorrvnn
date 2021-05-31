from functools import lru_cache
from matplotlib import colors

@lru_cache(maxsize=128)
def cachedPaths (doc) : 
    return doc.paths()

def pathColor (path, attr) : 
    if attr in path.element.attrib : 
        attr = colors.to_rgb(path.element.attrib[attr])
    else :
        attr = [1, 1, 1]
    return attr

def pathStrokeWidth (path) :
    if 'stroke-width' in path.element.attrib : 
        return float(path.element.attrib['stroke-width'])
    else :
        return 1
