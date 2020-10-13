import networkx as nx
from skimage import transform
import math
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
import matplotlib.pyplot as plt
from raster import svgStringToBitmap
from graphIO import GraphReadWrite
from imageio import imwrite

def treeAxisFromGraph(G, ax) : 
    pos = graphviz_layout(G, prog='dot')
    ax.set_aspect('equal')
    nx.draw(G, pos, ax=ax, node_size=0.5, arrowsize=1)
    for n in G :
        img = svgStringToBitmap(G.nodes[n]['svg'])
        imagebox = OffsetImage(img, zoom=0.1)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, pos[n], pad=0)
        ax.add_artist(ab)
    ax.axis('off')

def treeImageFromGraph (G) :
    """
    Visualize the paths as depicted
    by the tree structure of the 
    graph.

    This helps us understand whether the 
    network's decomposition is good or
    not.

    Parameters
    ----------
    G : nx.DiGraph
        Hierarchy of paths
    """
    fig, ax = plt.subplots(dpi=1500)
    pos = graphviz_layout(G, prog='dot')
    ax.set_aspect('equal')
    nx.draw(G, pos, ax=ax, node_size=0.5, arrowsize=1)
    for n in G :
        img = svgStringToBitmap(G.nodes[n]['svg'])
        imagebox = OffsetImage(img, zoom=0.2)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, pos[n], pad=0)
        ax.add_artist(ab)
    ax.axis('off')
    return (fig, ax)

def treeImageFromJson (jsonTuple) :
    """
    Given a tuple containing
    a single json file representing the
    tree structure of a particular svg,
    we visualize the figure using 
    matplotlib.

    For each rooted subtree, we keep an
    icon depicting the paths in that 
    subtree.

    This helps us understand whether the 
    network's decomposition is good or
    not.

    Parameters
    ----------
    jsonTuple : tuple
        singleton tuple containing path 
        to the json file containing
        tree data.
    """
    jsonFile, = jsonTuple
    G = GraphReadWrite('tree').read(jsonFile)
    return treeImageFromGraph(G)

def putOnCanvas (pts, images, outFile) :
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
    outFile : str
        Path to output file.
    """
    min_x = np.min(pts[:,0])
    max_x = np.max(pts[:,0])
    min_y = np.min(pts[:,1])
    max_y = np.max(pts[:,1])
    h, w, _ = images[0].shape
    sz = 10000
    pad = (h + w)
    pix = sz + 2 * pad
    canvas = np.ones((pix, pix, 4), dtype=np.uint8) * int(255)
    canvas[:, :, 3] = 0
    for pt, im in zip(pts, images) : 
        h_, w_, _ = im.shape
        if h_ != h or w_ != w :
            im = transform.resize(im, (h, w))
        pix_x = pad + math.floor(sz * ((pt[0] - min_x) / (max_x - min_x)))
        pix_y = pad + math.floor(sz * ((pt[1] - min_y) / (max_y - min_y)))
        sx, ex = pix_x - (h // 2), pix_x + (h // 2)
        sy, ey = pix_y - (w // 2), pix_y + (w // 2)
        canvas[sx:ex, sy:ey,:] = im
    imwrite(outFile, canvas)

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

if __name__ == "__main__" :
    import svgpathtools as svg
    from svgIO import setSVGAttributes
    from Dataset import SVGDataSet
    dataset = SVGDataSet('/Users/amaltaas/BTP/vectorrvnn/ManuallyAnnotatedDataset/CV', 'adjGraph', 10)
    doc = svg.Document(dataset[0].svgFile)
    paths = doc.flatten_all_paths()
    vb = doc.get_viewbox()
    setSVGAttributes(dataset[0], paths, vb)
    doc = svg.Document(dataset[1].svgFile)
    paths = doc.flatten_all_paths()
    vb = doc.get_viewbox()
    setSVGAttributes(dataset[1], paths, vb)
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=1500)
    treeAxisFromGraph(dataset[0], ax1)
    treeAxisFromGraph(dataset[1], ax2)
    ax1.set_title("og")
    ax2.set_title("inf")
    fig.savefig('out.png')
    plt.close(fig)
