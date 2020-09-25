import numpy as np
from treeOps import removeOneOutDegreeNodesFromTree, leaves, findRoot
import networkx as nx
import json
import os
import os.path as osp
from shapely.geometry import Polygon
import itertools
from functools import reduce
from scipy.spatial import ConvexHull
import svgpathtools as svg
import pymesh

def normalizePointCloud (ptCloud, height, width) :
    """
    Normalize a point cloud to fit into a box with 
    the top right corner at (width, height) and 
    bottom left corner at (0, 0).

    Parameters
    ----------
    ptCloud : numpy.ndarray
        N by 2 array of points.
    height : float
        Height of the viewbox.
    width : float
        Width of the viewbox.
    """
    xMax = ptCloud[:, 0].max()
    xMin = ptCloud[:, 0].min()
    yMax = ptCloud[:, 1].max()
    yMin = ptCloud[:, 1].min()

    scale = (0.8 * min(width, height)) / max(xMax - xMin, yMax - yMin)
    center = np.array([(xMax + xMin) / 2, (yMax + yMin) / 2])

    ptCloud -= center
    ptCloud *= scale
    ptCloud += np.array([width / 2, height / 2])
    return ptCloud

def normalizeTransform (ptCloud, width, height) :
    """
    Return a transform function to normalize some 
    subset of the point cloud.

    Parameters
    ----------
    ptCloud : numpy.ndarray
        N by 2 array of points. This is the
        original point cloud on the basis
        of which the transformation is calculated.
    height : float
        Height of the viewbox.
    width : float
        Width of the viewbox.

    """
    xMax = ptCloud[:, 0].max()
    xMin = ptCloud[:, 0].min()
    yMax = ptCloud[:, 1].max()
    yMin = ptCloud[:, 1].min()

    scale = (0.8 * min(width, height)) / max(xMax - xMin, yMax - yMin)
    center = np.array([(xMax + xMin) / 2, (yMax + yMin) / 2])
    newCenter = np.array([width / 2, height / 2])

    return lambda pt : scale * (pt - center) + newCenter

def Ry (theta) :
    """
    Return the 3D transformation matrix 
    for rotation around y-axis.

    Parameters
    ----------
    theta : float
        Angle of rotation in radians
    """
    return np.array([[np.cos(theta) , 0, np.sin(theta)], 
                     [0             , 1,             0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rx (theta) :
    """
    Return the 3D transformation matrix 
    for rotation around x-axis.

    Parameters
    ----------
    theta : float
        Angle of rotation in radians
    """
    return np.array([[1,             0,              0], 
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])

def Rz (theta) :
    """
    Return the 3D transformation matrix 
    for rotation around z-axis.

    Parameters
    ----------
    theta : float
        Angle of rotation in radians
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0], 
                     [np.sin(theta),  np.cos(theta), 0],
                     [0            ,              0, 1]])

def mesh2SVG(meshDir, xTheta, yTheta) :
    """
    Convert the mesh that is part of the PartNet
    dataset into a vector graphic document.

    The angles specify the camera position.

    Examples
    --------
    This code was used to generate the synthetic 
    dataset.

    >>> import sys
    >>> dirname = sys.argv[1]
    >>> yThetas = np.linspace(np.pi/4, 2 * np.pi, 4)
    >>> xThetas = [-np.pi/6, np.pi/6]
    >>> for name in tqdm(listdir(dirname)) : 
    >>>     _, number = osp.split(name)
    >>>     for xTheta, yTheta in itertools.product(xThetas, yThetas) :
    >>>         doc = mesh2SVG(name, xTheta, yTheta)
    >>>         filename = f'./PartNetVectorized/{number}-{xTheta}-{yTheta}.svg' 
    >>>         doc.save(filename)

    Parameters
    ----------
    meshDir : str
        Directory path which contains the partnet 
        data sample.
    xTheta : float
        Angle by which to rotate the x-axis.
    yTheta : float
        Angle by which to rotate the y-axis.
    """
    def sortedChildren(tree1, node, tree2) :
        if tree1.out_degree(node) != 0: 
            children = list(tree1.neighbors(node))
            for c in children : 
                sortedChildren(tree1, c, tree2)
            children.sort(key=lambda x: tree1.nodes[x]['key'])
            tree2.add_edges_from(list(itertools.product([node], children)))
            tree1.nodes[node]['key'] = tree1.nodes[children[-1]]['key'] 
        else : 
            tree2.add_node(node)
            tree2.nodes[node]['path'] = tree1.nodes[node]['path']

    def simplifyPolygon (pts, alpha) : 
        polygon = Polygon(pts).simplify(alpha, preserve_topology=False)
        pts_ = polygon.exterior.coords
        while len(pts_) == 0 : 
            polygon = Polygon(pts).simplify(alpha / 2, preserve_topology=False)
            pts_ = polygon.exterior.coords
            alpha = alpha / 2
        return pts_

    # This file contains the part hierarchy. The leaves contain the 
    # part objs that make up the 3D model.
    with open(osp.join(meshDir, 'result_after_merging.json')) as fd: 
        data = json.load(fd)
    hierarchy = nx.readwrite.json_graph.tree_graph(data[0])
    hierarchy = removeOneOutDegreeNodesFromTree(hierarchy)

    # Rotate around the y-axis by yTheta and then x-axis by xTheta.
    geomTransform = Rx(xTheta) @ Ry(yTheta)

    # We need to calculate how to make the projection of the
    # point cloud fit inside a viewbox. Here, we collect 
    # all the points and compute the transformation function
    # which normalizes the points so that they fit.
    pointCloud = []
    meshes = []
    for leaf in leaves(hierarchy) : 
        obj = hierarchy.nodes[leaf]['objs'][0] + '.obj'
        objpath = reduce(osp.join, (meshDir, 'objs', obj))
        mesh = pymesh.load_mesh(objpath)
        meshes.append(mesh)
        pointCloud.append(-mesh.vertices)
    pointCloud = ((geomTransform @ np.vstack(pointCloud).T).T)[:,:2]
    transform = normalizeTransform(pointCloud, 100, 100)
    # Now, we calculate the vector paths for each 
    # part in the 3D model.
    for leaf, mesh in zip(leaves(hierarchy), meshes) : 
        indices = mesh.faces.flatten()
        pts = -mesh.vertices[indices]
        pts = (geomTransform @ pts.T).T
        # Calculate the front most point of the projection for 
        # this part. Used to calculate the zIndex of the 
        # corresponding path. 
        hierarchy.nodes[leaf]['key'] = pts[:, 2].max()
        # Project the transformed points on the new X-Y axis.
        pts = pts[:, :2]
        # Perform the transform to fit viewbox.
        pts = transform(pts)
        # Compute the convex hull for the set of points.
        # Also, simplify the polygon using Douglas-Peucker.
        hull = ConvexHull(pts)
        pts = simplifyPolygon(pts[hull.vertices], 1)
        # Wrap around points for closed polygon.
        pts = np.vstack((pts, pts[0]))
        pts = [complex(*pt) for pt in pts]
        hierarchy.nodes[leaf]['path'] = svg.Path(*[svg.Line(a, b) for a, b in zip(pts, pts[1:])])

    document = svg.Document(None)
    document.set_viewbox('0 0 100 100')
    # Use the key property in the hierarchy to arrange the
    # children in the correct order.
    hierarchyWithCorrectOrder = nx.DiGraph()
    sortedChildren(hierarchy, findRoot(hierarchy), hierarchyWithCorrectOrder)
    # Add all the paths in the hierarchy into the document.
    document = tree2Document(document, hierarchyWithCorrectOrder, {"stroke": "black", "fill": "none"})
    return document
