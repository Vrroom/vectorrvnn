import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
from vectorrvnn.utils import *

def treeAxisFromGraph(G, ax) : 
    G_ = trimTreeByDepth(G, 4)
    doc = G_.doc
    pos = graphviz_layout(G_, prog='dot')
    ax.set_aspect('equal')
    nx.draw(G_, pos, ax=ax, node_size=0.5, arrowsize=1)
    md = max(1, np.ceil(maxDepth(G_) / 10))
    for n in G_ :
        subsetDoc = subsetSvg(doc, G_.nodes[n]['pathSet'])
        img = rasterize(subsetDoc, 128, 128)
        imagebox = OffsetImage(img, zoom=0.2 / md)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, pos[n], pad=0)
        ax.add_artist(ab)
    ax.axis('off')

def treeImageFromGraph (G) :
    """
    Visualize the paths as depicted by the tree structure 
    of the graph.  This helps us understand whether the 
    network's decomposition is good or not.

    Parameters
    ----------
    G : nx.DiGraph
        Hierarchy of paths
    """
    fig, ax = plt.subplots(dpi=100)
    treeAxisFromGraph(G, ax)
    return (fig, ax)

def putOnCanvas (pts, images) :
    """ 
    Put images at the corresponding
    points in R^2 in a super-large
    canvas.

    Mainly for tSNE visualization.
    
    The images should have an alpha
    channel.

    Parameters
    ----------
    pts : np.ndarray
        Collection of 2D points
    images : list
        Corresponding images to embed
    """
    min_x = np.min(pts[:,0])
    max_x = np.max(pts[:,0])
    min_y = np.min(pts[:,1])
    max_y = np.max(pts[:,1])
    h, w, _ = images[0].shape
    sz = 400
    pad = (h + w)
    pix = sz + 2 * pad
    canvas = np.ones((pix, pix, 4))
    canvas[:, :, 3] = 0
    for pt, im in zip(pts, images) : 
        h_, w_, _ = im.shape
        if h_ != h or w_ != w :
            im = transform.resize(im, (h, w))
        pix_x = int(pad + np.floor(sz * ((pt[0] - min_x) / (max_x - min_x))))
        pix_y = int(pad + np.floor(sz * ((pt[1] - min_y) / (max_y - min_y))))
        sx, ex = pix_x - (h // 2), pix_x + (h // 2)
        sy, ey = pix_y - (w // 2), pix_y + (w // 2)
        alpha = im[:, :, 3:]
        blob = canvas[sx:ex, sy:ey, :] 
        canvas[sx:ex, sy:ey,:] = np.clip(im * alpha + blob * (1 - alpha), 0, 1)
    canvas = alphaComposite(canvas, color=[0.5, 0.5, 0.5])
    return canvas

def matplotlibFigureSaver (obj, fname) :
    """ 
    Save matplotlib figures.

    Parameters
    ----------
    obj : tuple
        A tuple of plt.Figure, plt.Axes
    fname : str
        Path to output file.
    """
    fig, _ = obj
    fig.savefig(fname + '.png')
    plt.close(fig)

