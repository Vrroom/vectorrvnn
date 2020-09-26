# Avoid repetition at all costs.
import pickle
import os
import os.path as osp
import time
import json
import string
import subprocess
import more_itertools
import itertools
from functools import partial
from functools import reduce
from tqdm import tqdm
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
import copy
import multiprocessing as mp
import svgpathtools as svg
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import TextArea
from matplotlib.offsetbox import DrawingArea
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
from imageio import imwrite
import math
import Data
from scipy.spatial import ConvexHull, distance_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def area (bbox) : 
    """
    Compute the area of the bounding box.
    """
    bbox = bbox.view((-1, 4))
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

def iou (bbox1, bbox2) : 
    """
    Compute the intersection over union for two bounding
    boxes.

    Assume that there are N querying boxes and N target
    boxes. Hence the inputs have shape N x 4. The first
    two columns contain the coordinate of the top-left 
    corner and the last two contain the coordinates of
    the bottom-right corner.

    Parameters
    ----------
    bbox1 : torch.tensor
        Query bounding box.
    bbox2 : torch.tensor
        Target bounding box.
    """
    bbox1, bbox2 = bbox1.view((-1, 4)), bbox2.view((-1, 4))
    xMin, yMin = max(bbox1[:, 0], bbox2[:, 0]), max(bbox1[:, 1], bbox2[:, 1])
    xMax, yMax = min(bbox1[:, 2], bbox2[:, 2]), min(bbox1[:, 3], bbox2[:, 3])
    intersection = (xMax - xMin) * (yMax - yMin)
    union = area(bbox1) + area(bbox2)
    return intersection / union
    
def dictionaryProduct (d) : 
    product = itertools.product
    keys = d.keys()
    vals = map(lambda v : v if isinstance(v, list) else [v], d.values())
    combinations = product([keys], product(*vals))
    dicts = map(lambda items : dict(zip(*items)), combinations)
    return dicts

def aggregateDict (listOfDicts, reducer) : 
    keys = set(more_itertools.flatten(map(lambda x : x.keys(), listOfDicts)))
    aggregator = lambda key : reducer(map(lambda x : x[key], listOfDicts))
    return dict(zip(keys, map(aggregator, keys)))

def isBBoxDegenerate(bbox) :
    """
    Check if the bounding box has
    zero area.

    Parameters
    ----------
    bbox : tuple
        The top left and bottom right
        corners of the box.
    """
    xmin, xmax, ymin, ymax = bbox
    return xmin == xmax and ymin == ymax

def cluster2DiGraph (jsonFile, svgFile) :
    """
    Convert the output of the svg-app 
    clustering app into a digraph.
    """
    with open(jsonFile, 'r') as fd :
        data = json.load(fd)
    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    paths = list(filter(lambda p : not isBBoxDegenerate(p.path.bbox()), paths))
    tree = nx.DiGraph()
    tree.add_nodes_from(range(len(data["nodes"])))
    for node in data["nodes"] : 
        tree.nodes[node["id"]]["pathSet"] = node["paths"]
        for child in node["children"] :
            tree.add_edge(node["id"], child)
    return Data.Tree(tree)

class AllPathDescriptorFunction () : 
    """ 
    Given a descriptor function, convert it into
    one that acts on all paths
    """

    def __init__ (self, descFunction, **kwargs) : 
        self.descFunction = descFunction
        self.model = None
        if 'model' in kwargs : 
            self.model = kwargs['model']

    def __call__ (self, paths, vb) : 
        descriptors = [self.descFunction(p.path, vb) for p in paths]
        if self.model : 
            descriptors = torch.tensor(descriptors)
            edges = Pose.boneConnectivity
            pathTypes = [p.element.attrib['id'] for p in paths]
            pathTypeIdx = dict(map(reversed, enumerate(pathTypes)))
            targets = torch.tensor([pathTypeIdx[t] for t in pathTypes]).long()
            edges = [(pathTypeIdx[a], pathTypeIdx[b]) for a, b in edges]
            edges = torch.t(torch.tensor(edges).long())
            data = tgd.Data(x=descriptors, edge_index=edges, y=targets)
            return self.model.encoder(data.x, data.edge_index).detach().numpy()
        else :
            return np.array(descriptors)
