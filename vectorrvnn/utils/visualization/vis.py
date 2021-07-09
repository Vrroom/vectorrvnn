import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
from vectorrvnn.utils import *
from itertools import islice, cycle
from more_itertools import collapse

COLOR_MAP = dict(
    red=[1, 0, 0],
    green=[0, 1, 0],
    blue=[0, 0, 1],
    yellow=[1, 1, 0],
)

def treeAxisFromGraph(G, fig, ax) : 
    G.graph['nodesep'] = 1
    doc = G.doc
    pos = graphviz_layout(G, prog='dot')
    ax.set_aspect('equal')
    nx.draw(G, pos, ax=ax, node_size=0.5, arrowsize=1)
    # horizontal space for one tree in pixels
    pixX = fig.get_figwidth() * fig.dpi / 4
    # max number of images that'll be side by side
    sideBySideIms = maxNodesByLevel(G) + 1
    # raster size
    imSize = 128
    # zoom for each view
    zoom = pixX / (sideBySideIms * imSize * 3)
    for n in G :
        subsetDoc = subsetSvg(doc, G.nodes[n]['pathSet'])
        img = rasterize(subsetDoc, imSize, imSize)
        imagebox = OffsetImage(img, zoom=zoom)
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
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    treeAxisFromGraph(G, fig, ax)
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

def _xrange (pos) :
    xs = [v[0] for v in pos.values()]
    return min(xs), max(xs)

def _yrange (pos) :
    ys = [v[1] for v in pos.values()]
    return min(ys), max(ys)

def _calculateShift (pos1, pos2) :
    mx1, Mx1 = _xrange(pos1)
    mx2, Mx2 = _xrange(pos2)
    my1, My1 = _yrange(pos1)
    my2, My2 = _yrange(pos2)
    my = min(my1, my2)
    My = max(My1, My2)
    shift = max(Mx1 - mx2, 0) + ((Mx1 - mx1) + (Mx2 - mx2)) / 8
    pos2 = dictmap(lambda k, v : (v[0] + shift, v[1]), pos2)
    pos1 = dictmap(
        lambda k, v : (v[0], my + (My - my) * (v[1] - my1) / (My1 - my1)),
        pos1
    )
    pos2 = dictmap(
        lambda k, v : (v[0], my + (My - my) * (v[1] - my2) / (My2 - my2)),
        pos2
    )
    return pos1, pos2

def treeMatchVis (t1, t2, matchMatrix) :
    fig, ax = plt.subplots(figsize=(20, 5), dpi=200)
    treeMatchVisOnAxis(t1, t2, matchMatrix, fig, ax)
    return (fig, ax)

def treeMatchVisOnAxis (t1, t2, matchMatrix, fig, ax, prefix=('1-', '2-')) :
    t1.graph['nodesep'] = t2.graph['nodesep'] = 1
    pos1 = graphviz_layout(t1, prog='dot')
    pos2 = graphviz_layout(t2, prog='dot')
    pos1, pos2 = _calculateShift(pos1, pos2)

    nodes1, nodes2 = list(t1.nodes), list(t2.nodes)

    # set the match graph and mark positions.
    a, b = nx.DiGraph(t1), nx.DiGraph(t2)
    g = nx.union(a, b, rename=prefix)
    g.remove_edges_from(list(g.edges))
    gPos = dict()
    edge_color = list(islice(
        collapse(cycle([
            'red',
            'yellow',
            'green',
            'blue',
        ])),
        0,
        int((matchMatrix == 1).sum())
    ))
    for (i, j), color in zip(zip(*np.where(matchMatrix == 1)), edge_color) :
        u = f'{prefix[0]}{nodes1[i]}'
        v = f'{prefix[1]}{nodes2[j]}'
        g.add_edge(u, v)
        gPos[u] = pos1[nodes1[i]]
        gPos[v] = pos2[nodes2[j]]
        a.nodes[nodes1[i]]['color'] = color
        b.nodes[nodes2[j]]['color'] = color

    nx.draw_networkx_nodes(
        t1,
        pos1,
        ax=ax,
        node_size=0.5
    )
    nx.draw_networkx_nodes(
        t2,
        pos2,
        ax=ax,
        node_size=0.5
    )
    nx.draw_networkx_edges(
        t1,
        pos1,
        ax=ax,
        arrowsize=1
    )
    nx.draw_networkx_edges(
        t2,
        pos2,
        ax=ax,
        arrowsize=1
    )
    # horizontal space for one tree in pixels
    pixX = fig.get_figwidth() * fig.dpi / 4
    # max number of images that'll be side by side
    sideBySideIms = max(maxNodesByLevel(t1), maxNodesByLevel(t2)) + 1
    # raster size
    imSize = 128
    # zoom for each view
    zoom = pixX / (sideBySideIms * imSize * 3)
    for n in t1 :
        subsetDoc = subsetSvg(t1.doc, t1.nodes[n]['pathSet'])
        img = rasterize(subsetDoc, imSize, imSize)
        color = [1, 1, 1]
        if 'color' in a.nodes[n] :
            color = COLOR_MAP[a.nodes[n]['color']]
        img = alphaComposite(img, module=np, color=color)
        imagebox = OffsetImage(img, zoom=zoom)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, pos1[n], pad=0)
        ax.add_artist(ab)

    for n in t2 :
        subsetDoc = subsetSvg(t2.doc, t2.nodes[n]['pathSet'])
        img = rasterize(subsetDoc, imSize, imSize)
        color = [1, 1, 1]
        if 'color' in b.nodes[n] :
            color = COLOR_MAP[b.nodes[n]['color']]
        img = alphaComposite(img, module=np, color=color)
        imagebox = OffsetImage(img, zoom=zoom)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, pos2[n], pad=0)
        ax.add_artist(ab)

    ax.axis('off')


