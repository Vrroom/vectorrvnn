import subprocess
from copy import deepcopy
import numpy as np
import random
import os
import os.path as osp
import svgpathtools as svg
import matplotlib.image as image
import string

def randomString(k) : 
    """
    Return a random string of lower case
    alphabets.

    Parameters
    ----------
    k : int
        Length of the random string.
    """
    alphabets = string.ascii_lowercase
    return ''.join(random.choices(alphabets, k=k))

def rasterize(svgFile, outFile, H=72, W=72, makeSquare=True) :
    """
    Utility function to convert svg to
    raster using inkscape

    Parameters
    ----------
    svgFile : str
        Path to input svg
    outFile : str
        Path to where to store output
    """
    with open(os.devnull, 'w') as fd : 
        if makeSquare : 
            subprocess.call(['inkscape', f'-h {H}', f'-w {W}', '-z', '-f', svgFile, '-j', '-e', outFile], stdout=fd)
        else : 
            subprocess.call(['inkscape', '-z', '-f', svgFile, '-j', '-e', outFile], stdout=fd)

def singlePathSvg(path, vb, out) :
    """
    Write a single path to svg file.

    Parameters
    ----------
    path : svg.Document.FlattenedPath
        Path to be written
    vb : list
        The viewbox of the svg
    out : str
        Path to the output file.
    """
    vbox = ' '.join([str(_) for _ in vb])
    svg.wsvg([path[0]], attributes=[path[1].attrib], viewbox=vbox, filename=out)

def getSubsetSvg(paths, lst, vb) :
    """
    An svg is a collection of paths. 
    This function chooses a list of
    paths from the svg and makes an svg
    string for those paths. 

    While doing this, we have to be careful
    about the order in which these paths
    are put because they determine the
    order in which they are rendered. 

    For this, I have tweaked the 
    svgpathtools library to also
    store the zIndex of the paths
    so that we have help while putting
    the paths together

    >>> doc = svg.Document('file.svg')
    >>> paths = doc.flatten_all_paths()
    >>> print(getSubsetSvg(paths, [1,2,3], doc.get_viewbox()))

    Parameters
    ----------
    paths : list
        List where each element is of
        the type svg.Document.FlattenedPath
    lst : list
        An index set into the previous list 
        specifying the paths we want.
    vb : list
        The viewbox of the original svg.
    """
    vbox = ' '.join([str(_) for _ in vb])
    ps = [p[0] for p in paths]
    # opacity = {"fill-opacity": 0.2, "stroke-opacity": 0.2}
    attrs = [deepcopy(p[1].attrib) for p in paths]
    # for i in range(len(paths)) :
    #     if i not in lst: 
    #         attrs[i].update(opacity)
    order = [p[3] for p in paths]
    cmb = list(zip(order, ps, attrs))
    cmb.sort()
    ps = [p for _, p, _ in cmb]
    attrs = [a for _, _, a in cmb]
    drawing = svg.disvg(ps, 
                        attributes=attrs, 
                        viewbox=vbox, 
                        paths2Drawing=True, 
                        openinbrowser=False)
    return drawing.tostring()

def getSubsetSvg2(paths, lst, vb) :
    vbox = ' '.join([str(_) for _ in vb])
    ps = [paths[i][0] for i in lst]
    attrs = [deepcopy(paths[i][1].attrib) for i in lst]
    order = [paths[i][3] for i in lst]
    cmb = list(zip(order, ps, attrs))
    cmb.sort()
    ps = [p for _, p, _ in cmb]
    attrs = [a for _, _, a in cmb]
    drawing = svg.disvg(ps, 
                        attributes=attrs, 
                        viewbox=vbox, 
                        paths2Drawing=True, 
                        openinbrowser=False)
    return drawing.tostring()

def alphaCompositeOnWhite (source) : 
    destination = np.ones_like(source)
    alpha = source[:, :, 3:]
    d_ = destination[:, :, :3]
    s_ = source[:, :, :3]
    return d_ * (1 - alpha) + s_ * alpha

def svgStringToBitmap (svgString, H, W) :
    svgName = randomString(10) + '.svg'
    svgName = osp.join('/tmp', svgName)
    pngName = randomString(10) + '.png'
    pngName = osp.join('/tmp', pngName)
    with open(svgName, 'w+') as fd :
        fd.write(svgString)
    rasterize(svgName, pngName, H, W)
    img = image.imread(pngName)
    os.remove(svgName)
    os.remove(pngName)
    return alphaCompositeOnWhite(img)

def SVGSubset2NumpyImage (doc, pathSet, H, W) :
    paths = doc.flatten_all_paths()
    boxes = np.array([paths[i].path.bbox() for i in pathSet])
    docBox = doc.get_viewbox()
    docDim = min(docBox[2], docBox[3])
    eps = docDim / 20
    xm, xM = boxes[:,0].min(), boxes[:,1].max()
    ym, yM = boxes[:,2].min(), boxes[:,3].max()
    h, w = xM - xm, yM - ym
    d = max(h, w)
    if h > w : 
        box = [xm, ym+w/2-d/2, d, d]
    else : 
        box = [xm+h/2-d/2, ym, d, d]
    box[0] -= eps
    box[1] -= eps
    box[2] += 2 * eps
    box[3] += 2 * eps
    svgString = getSubsetSvg(paths, pathSet, box)
    return svgStringToBitmap(svgString, H, W)

def SVGSubset2NumpyImage2 (doc, pathSet, H, W) :
    paths = doc.flatten_all_paths()
    boxes = np.array([paths[i].path.bbox() for i in pathSet])
    docBox = doc.get_viewbox()
    svgString = getSubsetSvg2(paths, pathSet, docBox)
    return svgStringToBitmap(svgString, H, W)

def SVGtoNumpyImage (svgFilePath, H, W) :
    """
    Take an SVG file and rasterize it to 
    obtain a numpy array of given height and
    width. 

    Parameters
    ----------
    svgFilePath : str
    H : float
        Desired height of output.
    W : float 
        Desired width of output.
    """
    with open(svgFilePath) as fd : 
        string = fd.read()
    return svgStringToBitmap(string, H, W)
