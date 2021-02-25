import networkx as nx
import os
import os.path as osp
import torch
from Dataset import SVGDataSet
from tqdm import tqdm
import json
from skimage import transform
import math
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
import matplotlib.pyplot as plt
from raster import svgStringToBitmap, alphaCompositeOnGrey, alphaCompositeOnWhite
from graphIO import GraphReadWrite
from imageio import imwrite

def treeAxisFromGraph(G, ax) : 
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
        img = svgStringToBitmap(G.nodes[n]['svg'], 64, 64)
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
    sz = 500
    pad = (h + w)
    pix = sz + 2 * pad
    canvas = np.ones((pix, pix, 4))#, dtype=np.uint8) * int(255)
    canvas[:, :, 3] = 0
    for pt, im in zip(pts, images) : 
        h_, w_, _ = im.shape
        if h_ != h or w_ != w :
            im = transform.resize(im, (h, w))
        pix_x = pad + math.floor(sz * ((pt[0] - min_x) / (max_x - min_x)))
        pix_y = pad + math.floor(sz * ((pt[1] - min_y) / (max_y - min_y)))
        sx, ex = pix_x - (h // 2), pix_x + (h // 2)
        sy, ey = pix_y - (w // 2), pix_y + (w // 2)
        alpha = im[:, :, 3:]
        blob = canvas[sx:ex, sy:ey, :] 
        canvas[sx:ex, sy:ey,:] = np.clip(im * alpha + blob * (1 - alpha), 0, 1)

    canvas = alphaCompositeOnGrey(canvas)
    plt.imsave(outFile, canvas, format='png')

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

def violinPlot (path_vae_name, name, config, useColor, nMerge) :
    def collate_fn (batch) : 
        allLeaf = lambda n, t : set(leaves(t)).issuperset(set(t.neighbors(n)))
        paths = []
        numNeighbors = []
        for t in batch : 
            for n in nonLeaves(t) : 
                if allLeaf(n, t) : 
                    if t.out_degree(n) == nMerge : 
                        paths.append(torch.stack([t._path(_) for _ in t.neighbors(n)]))
                        numNeighbors.append(t.out_degree(n))
        paths = nn.utils.rnn.pad_sequence(paths, batch_first=True)
        ret = dict(paths=paths, numNeighbors=numNeighbors)
        return ret
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load test data
    testDir = commonConfig['test_directory']
    testData = SVGDataSet(testDir, 'adjGraph', 10, useColor=useColor)
    testData.toTensor()
    # Load model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VAE_OUTPUT = os.path.join(BASE_DIR, "results", path_vae_name)
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", name)
    model = OneMergeAutoEncoder(PathVAE(config), config['sampler'])
    interface = MergeInterface(model)
    state_dict = torch.load(os.path.join(MERGE_OUTPUT, 'training_end.pth'))
    model.load_state_dict(state_dict['model'])
    batch = collate_fn(testData)
    fwd_data = interface.forward(batch)
    reconLosses = interface._individualLosses(batch, fwd_data)
    randomLosses = []
    for i in range(len(reconLosses)) : 
        example = random.choice(testData)
        n = nMerge
        perm = torch.randperm(example.descriptors.size(0))
        idx = perm[:n]
        randomMerge = example.descriptors[idx]
        randomMerge = randomMerge.unsqueeze(0) 
        batch = dict(paths=randomMerge, numNeighbors=[n])
        fwd_data = interface.forward(batch)
        losses = interface._individualLosses(batch, fwd_data)
        randomLosses.extend(losses)
    data = [reconLosses, randomLosses]
    return data

