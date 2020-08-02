# Avoid repetition at all costs.
import pickle
import os
import os.path as osp
from pulp import LpVariable
import time
import json
import string
import subprocess
import more_itertools
import itertools
from functools import reduce
import networkx as nx
import xml.etree.ElementTree as ET
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms import bipartite
from networkx.algorithms.community import kernighan_lin_bisection
import copy
import multiprocessing as mp
import svgpathtools as svg
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea
from matplotlib.offsetbox import DrawingArea
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
from imageio import imwrite
import math
from skimage import transform
import Data
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import scipy.io as sio

class Pose () :

    joints = {
        0: 'r_ankle', 1: 'r_knee', 2: 'r_hip',
        3: 'l_hip', 4: 'l_knee', 5: 'l_ankle',
        6: 'pelvis', 7: 'thorax', 8: 'upper_neck',
        9: 'head_top', 10: 'r_wrist', 11: 'r_elbow',
        12: 'r_shoulder', 13: 'l_shoulder', 14: 'l_elbow',
        15: 'l_wrist'
    }

    bones = {
        'r_thigh': (1, 2),
        'r_calf': (0, 1),
        'l_thigh': (3, 4),
        'l_calf': (4, 5),
        'r_upper_arm': (12, 11),
        'l_upper_arm': (13, 14),
        'r_lower_arm': (11, 10),
        'l_lower_arm': (14, 15),
        'torso': (13, 3, 6, 2, 12, 7),
        'neck': (7, 8),
        'head': (8, 9)
    }
    
    def __init__ (self, pose) :
        self.xNoise, self.yNoise = PerlinNoise(seed=0), PerlinNoise(seed=1)
        self.pose = np.array(pose)
        revMap = dict(zip(self.joints.values(), self.joints.keys()))
        self.joints.update(revMap)
        self._normalize()
        self._createConnectivityGraph()
        self._createPathHierarchy()
        self._createBonePaths()

    def _np2Complex(self, p) :
        return complex(p[0], p[1])

    def _complex2np(self, p) :
        return np.array([p.real, p.imag])

    def _headPath (self) :
        p1, p2 = [self.pose[i] for i in self.bones['head']]
        r = np.linalg.norm(p2 - p1) / 2
        center = self._np2Complex((p1 + p2)/2)
        return noisyPath(circle(r).translated(center))

    def _torsoPath (self) :
        points = [self.pose[i] for i in self.bones['torso']]
        points = [self._np2Complex(p) for p in points]
        points = [p + self._noise(p) for p in points]
        return smoothSpline(points)

    def _rectPath (self, p1, p2) :
        delta = np.flip(p2 - p1)
        delta[0] = -delta[0]
        d = np.linalg.norm(delta)
        delta /= 4
        c1, c2 = self._np2Complex(p1 + delta), self._np2Complex(p1 - delta)
        c3, c4 = self._np2Complex(p2 + delta), self._np2Complex(p2 - delta)
        l1, l2 = svg.Line(c1, c2), svg.Line(c2, c4)
        l3, l4 = svg.Line(c4, c3), svg.Line(c3, c1)
        path = svg.Path(l1, l2, l3, l4)
        return noisyPath(path)

    def _normalize (self) : 
        xMax = self.pose[:, 0].max()
        xMin = self.pose[:, 0].min()
        yMax = self.pose[:, 1].max()
        yMin = self.pose[:, 1].min()

        scale = 80 / max(xMax - xMin, yMax - yMin)
        center = np.array([(xMax + xMin) / 2, (yMax + yMin) / 2])

        self.pose -= center
        self.pose *= scale
        self.pose += np.array([50, 50])

    def _createConnectivityGraph (self) :
        self.G = nx.Graph() 
        self.G.add_edges_from([
            (0, 1), (1, 2), (3, 4),
            (4, 5), (2, 6), (3, 6),
            (6, 7), (7, 8), (8, 9),
            (7, 12), (7, 13), (13, 14),
            (14, 15), (12, 11), (11, 10)
        ])

    def _createPathHierarchy (self) :
        self.T = nx.DiGraph()
        self.T.add_edges_from([
            ('person', 'upper_body'), ('person', 'limbs'),
            ('upper_body', 'torso'), ('upper_body', 'neck&Head'),
            ('neck&Head', 'head'), ('neck&Head', 'neck'),
            ('limbs', 'legs'), ('limbs', 'arms'),
            ('legs', 'l_leg'), ('legs', 'r_leg'),
            ('arms', 'l_arm'), ('arms', 'r_arm'),
            ('l_arm', 'l_upper_arm'), ('l_arm', 'l_lower_arm'),
            ('r_arm', 'r_upper_arm'), ('r_arm', 'r_lower_arm'),
            ('l_leg', 'l_thigh'), ('l_leg', 'l_calf'),
            ('r_leg', 'r_thigh'), ('r_leg', 'r_calf'),
        ])

    def visualizePose (self) : 
        plt.gca().set_aspect('equal')
        nx.draw(self.G, self.pose)
        plt.show()

    def _noise (self, p) :
        return complex(self.xNoise(p), self.yNoise(p))

    def _npToComplex (self, listOfVectors) :
        return [complex(p[0], p[1]) for p in listOfVectors]

    def _createBonePaths (self) : 
        self.paths = {'head': self._headPath(), 'torso': self._torsoPath()}
        for k, v in self.bones.items() :
            if k != 'head' and k != 'torso': 
                p1, p2 = [self.pose[i] for i in v]
                self.paths[k] = self._rectPath(p1, p2)

    def getDocument (self) :

        def addToDocument (T, r, children) :
            if T.out_degree(r) == 0 : 
                doc.add_path(self.paths[r], group=parentGroup[r])
            else :
                newGroup = doc.add_group(parent=parentGroup[r], group_attribs={"fill":"black", "fill-opacity":"0.5"})
                for child in children: 
                    parentGroup[child] = newGroup

        doc = svg.Document(None)
        doc.set_viewbox('0 0 100 100')
        parentGroup = {'person': None}
        treeApplyRootFirst(self.T, 'person', addToDocument)
        return doc

def noisyPath (path) : 
    """
    Add Perlin Noise to path.

    Right now, I've done something which works well. 
    I perturb the path at 10 evenly spaced points. Then,
    I interpolate a smooth spline over them.

    Parameters
    ----------
    path: svg.Path
        Path to which noise has to be added.
    """
    xNoise, yNoise = PerlinNoise(seed=0), PerlinNoise(seed=1)
    points = [path.point(t) for t in np.arange(0, 1.01, 0.1)]
    points = [p + complex(xNoise(p), yNoise(p)) for p in points]
    return smoothSpline(points)

def rectangle(h, w) :
    rect = []
    rect.append(svg.Line(0 + 0j, w + 0j))
    rect.append(svg.Line(w + 0j, w + h * 1j))
    rect.append(svg.Line(w + h * 1j, 0 + h * 1j))
    rect.append(svg.Line(0 + h * 1j, 0 + 0j))
    return svg.Path(*rect)

def circle (r) :
    arcs = []
    arcs.append(svg.Arc(-r + 0j, r + r * 1j, 0, False, False, r + 0j))
    arcs.append(svg.Arc(r + 0j, r + r * 1j, 0, False, False, -r + 0j))
    path = svg.Path(*arcs)
    return path

class Person () :

    def __init__ (self) :
        self.center = 50 + 50j
        headRotation = random.randint(-30, 30)
        torsoRotation = random.randint(-15, 15)
        h, w = random.randint(20, 35), random.randint(15, 25)
        diff = self.center - (w + (h * 1j))/2
        self.torso = Torso(h, w).translated(diff).rotated(torsoRotation)
        self.head = Head(random.randint(5, 10)).rotated(headRotation)
        self.arms = [Limb(*self.limbDimensions()).rotated(random.randint(135, 225)).bottomRotated(random.randint(-30, 30)),
                     Limb(*self.limbDimensions()).rotated(random.randint(-45, 45)).bottomRotated(random.randint(-30, 30))]
        self.legs = [Limb(*self.limbDimensions()).rotated(random.randint(70, 110)).bottomRotated(random.randint(-30, 30)),
                     Limb(*self.limbDimensions()).rotated(random.randint(70, 110)).bottomRotated(random.randint(-30, 30))]
        self.attach()

    def limbDimensions (self) :
        h1 = random.randint(5, 10)
        h2 = random.randint(5, 10)
        w1 = random.randint(15, 20)
        w2 = random.randint(15, 20)
        return h1, w1, h2, w2

    def attach (self) :
        self.head = self.head.translated(self.torso.joints[-1] - self.head.joint)
        self.arms = [a.translated(self.torso.joints[i] - a.joint) for i, a in enumerate(self.arms)]
        self.legs = [l.translated(self.torso.joints[2 + i] - l.joint) for i, l in enumerate(self.legs)]

    def addToDocument (self, doc) :
        top = doc.add_group(group_attribs={"fill":"black"})
        body = doc.add_group(parent=top)
        limbs = doc.add_group(parent=top)
        legsGroup = doc.add_group(parent=limbs)
        armsGroup = doc.add_group(parent=limbs)
        bothLegGroups = [doc.add_group(parent=legsGroup) for _ in range(2)]
        bothArmGroups = [doc.add_group(parent=armsGroup) for _ in range(2)]
        doc.add_path(noisyPath(self.head.circle), group=body)
        doc.add_path(noisyPath(self.torso.body), group=body)

        for element, group in zip(self.legs, bothLegGroups) :
            doc.add_path(noisyPath(element.top), group=group)
            doc.add_path(noisyPath(element.bottom), group=group)

        for element, group in zip(self.arms, bothArmGroups) :
            doc.add_path(noisyPath(element.top), group=group)
            doc.add_path(noisyPath(element.bottom), group=group)

        return doc

class Torso () :
    
    def __init__ (self, h, w) :
        self.body = rectangle(h, w)
        self.joints = [
            (h/4) * 1j,
            w + (h/4) * 1j,
            (w/4) + h * 1j,
            (3*w/4) + h * 1j,
            w/2
        ]
        self.center = (w + h*1j) / 2
        self.pseudoLines = [svg.Line(start=self.center, end=p) for p in self.joints]

    def translated(self, t) :
        newTorso = copy.deepcopy(self)
        newTorso.body = newTorso.body.translated(t)
        newTorso.joints = [j + t for j in newTorso.joints]
        newTorso.center += t
        newTorso.pseudoLines = [l.translated(t) for l in newTorso.pseudoLines]
        return newTorso

    def rotated(self, angle) :
        newTorso = copy.deepcopy(self)
        newTorso.body = newTorso.body.rotated(angle, newTorso.center)
        newTorso.pseudoLines = [l.rotated(angle, self.center) for l in newTorso.pseudoLines]
        newTorso.joints = [l.end for l in newTorso.pseudoLines]
        return newTorso

class Limb () : 

    def __init__ (self, h1, w1, h2, w2) :
        self.top = rectangle(h1, w1)
        self.joint = (h1/2) * 1j
        self.joint2 = w1 + (h1/2) * 1j
        self.bottom = rectangle(h2, w2).translated(self.joint2 - (h2/2) * 1j)
        self.pseudoLine = svg.Line(start=self.joint, end=self.joint2)

    def translated(self, t) : 
        newLimb = copy.deepcopy(self)
        newLimb.top = newLimb.top.translated(t)
        newLimb.bottom = newLimb.bottom.translated(t)
        newLimb.joint += t
        newLimb.joint2 += t
        return newLimb

    def rotated (self, angle) :
        newLimb = copy.deepcopy(self)
        newLimb.top = newLimb.top.rotated(angle, newLimb.joint)
        newLimb.bottom = newLimb.bottom.rotated(angle, newLimb.joint)
        newLimb.pseudoLine = newLimb.pseudoLine.rotated(angle, newLimb.joint)
        newLimb.joint2 = newLimb.pseudoLine.end
        return newLimb

    def bottomRotated (self, angle) :
        newLimb = copy.deepcopy(self)
        newLimb.bottom = newLimb.bottom.rotated(angle, newLimb.joint2)
        return newLimb

class Head () :

    def __init__ (self, r) :
        self.circle = circle(r)
        self.joint = r * 1j

    def translated (self, t) :
        newHead = copy.deepcopy(self)
        newHead.circle = newHead.circle.translated(t)
        newHead.joint += t
        return newHead

    def rotated (self, angle) :
        newHead = copy.deepcopy(self)
        newHead.circle = newHead.circle.rotated(angle, newHead.joint)
        return newHead


def constructLinearSystem (constraints, variables,
    matrixConstructor, vectorConstructor) :

    def insert (d, r, c) : 
        data.append(d)
        rows.append(r)
        cols.append(c)

    data, rows, cols = [], [], []
    varIdx = dict(zip(variables, range(len(variables))))
    b = []
    for i, constraint in enumerate(constraints) :
        for v in constraint.keys() :
            insert(constraint[v], i, varIdx[v])
        b.append(-constraint.constant)
    A = matrixConstructor((data, (rows, cols)))
    b = vectorConstructor(b)
    return A, b

def smoothSpline(points) :
    """
    Create a smooth spline over a set of points. Method
    obtained from:

        https://www.particleincell.com/2012/bezier-splines/

    Examples
    --------
    >>> points = [complex(0, 0), complex(10, 20), complex(20, 20)]
    >>> path = smoothSpline(points)
    >>> points = [path.point(t) for t in np.arange(0, 1, 0.1)]
    >>> x = [p.real for p in points]
    >>> y = [p.imag for p in points]
    >>> plt.plot(x, y)
    >>> plt.show()
    
    Parameters
    ----------
    points: list
        List of points in the complex plane where
        the real coordinate is the x-coordinate 
        and the imaginary coordinate is the y-coordinate
    """
    def linearSystem() : 
        # Prepare linear system of equations.
        eqns = []
        for i in range(n - 1) :
            # First Constraint: P_(i + 1)_1 + P_i_2 = 2K_(i + 1)
            v1x, v2x = f'P_{i + 1}_1_x', f'P_{i}_2_x'
            v1y, v2y = f'P_{i + 1}_1_y', f'P_{i}_2_y'
            ki_1 = points[i+1]
            eqns.append(V[v1x] + V[v2x] == 2 * ki_1.real)
            eqns.append(V[v1y] + V[v2y] == 2 * ki_1.imag)

            # Second Constraint:
            # -2P_(i + 1)_1 + P_(i + 1)_2 - P_i_1 + 2P_i_2 = 0
            v3x, v4x = f'P_{i + 1}_2_x', f'P_{i}_1_x'
            v3y, v4y = f'P_{i + 1}_2_y', f'P_{i}_1_y'
            eqns.append(-2*V[v1x] + V[v3x] - V[v4x] + 2*V[v2x] == 0)
            eqns.append(-2*V[v1y] + V[v3y] - V[v4y] + 2*V[v2y] == 0)

        if abs(points[0] - points[-1]) < 1e-3 : 
            # First Constraint: P_(n - 1)_1 + P_0_2 = 2K_(0)
            v1x, v2x = f'P_0_1_x', f'P_{n-1}_2_x'
            v1y, v2y = f'P_0_1_y', f'P_{n-1}_2_y'
            k0 = points[0]
            eqns.append(V[v1x] + V[v2x] == 2 * k0.real)
            eqns.append(V[v1y] + V[v2y] == 2 * k0.imag)

            # Second Constraint:
            # -2P_0_1 + P_0_2 - P_(n - 1)_1 + 2P_(n - 1)_2 = 0
            v3x, v4x = f'P_0_2_x', f'P_{n-1}_1_x'
            v3y, v4y = f'P_0_2_y', f'P_{n-1}_1_y'
            eqns.append(-2*V[v1x] + V[v3x] - V[v4x] + 2*V[v2x] == 0)
            eqns.append(-2*V[v1y] + V[v3y] - V[v4y] + 2*V[v2y] == 0)
        else : 
            # 4 Boundary Condition constraints for open paths
            # 2P_0_1 - P_0_2 = K_0
            k0 = points[0]
            eqns.append(2*V['P_0_1_x'] - V['P_0_2_x'] == k0.real)
            eqns.append(2*V['P_0_1_y'] - V['P_0_2_y'] == k0.imag)

            # 2P_(n - 1)_2 - P_(n - 1)_1 = K_n
            kn = points[-1]
            eqns.append(2*V[f'P_{n-1}_2_x'] - V[f'P_{n-1}_1_x'] == kn.real)
            eqns.append(2*V[f'P_{n-1}_2_y'] - V[f'P_{n-1}_1_y'] == kn.imag)
        
        return constructLinearSystem(eqns, V.values(), csc_matrix, np.array)

    def makePathFromSolutionVector (x) :
        beziers = []
        for i, knots in enumerate(zip(points, points[1:])) :
            start, end = knots
            P_i_1_x, P_i_1_y = V[f'P_{i}_1_x'], V[f'P_{i}_1_y']
            P_i_2_x, P_i_2_y = V[f'P_{i}_2_x'], V[f'P_{i}_2_y']
            cp1 = complex(x[VIdx[P_i_1_x]], x[VIdx[P_i_1_y]])
            cp2 = complex(x[VIdx[P_i_2_x]], x[VIdx[P_i_2_y]])
            beziers.append(svg.CubicBezier(start, cp1, cp2, end))
        return svg.Path(*beziers)
    
    def initVariables () :
        for i in range(n) : 
            for j in range(1, 3) : 
                xName, yName = f'P_{i}_{j}_x', f'P_{i}_{j}_y'
                V[xName] = LpVariable(xName)
                V[yName] = LpVariable(yName)

    n = len(points) - 1
    V = dict()
    initVariables()
    VIdx = dict(zip(V.values(), range(len(V))))
    A, b = linearSystem()
    x = spsolve(A, b)
    return makePathFromSolutionVector(x)

class PerlinNoise () :
    """ 
    Implementation of the Perlin Method mentioned at:

        http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf

    to generate well behaved noise in 2D plane

    Examples
    --------
    >>> noise = PerlinNoise(seed=10)
    >>> image = np.zeros((100, 100))
    >>> for i in range(100) :
    >>>     for j in range(100) :
    >>>         image[i, j] = noise(complex(i / 10, j / 10))
    >>> plt.imshow(image)
    >>> plt.show()
    """

    nGrads = 16

    def __init__ (self, seed=0) :
        """
        Constructor.

        Fix the gradients for the lattice points.
        """
        rng = np.random.RandomState(seed)
        self.grads = [self._angleToGrad(rng.uniform(high=2*np.pi)) for _ in range(self.nGrads)]

    def _gradIdx (self, a, b) :
        n = (a * b) % self.nGrads
        return n
    
    def _angleToGrad (self, angle) :
        return complex(np.cos(angle), np.sin(angle))

    def _lattice (self) :
        points = itertools.product(range(2), range(2))
        return points

    def _f (self, t) : 
        return (6 * (t ** 5)) - (15 * (t ** 4)) + (10 * (t ** 3))

    def __call__ (self, p) :
        """
        Given a point somewhere in the 2D plane, 
        find out how much noise is to be added to that point.
        """
        x, y = p.real, p.imag
        if (x.is_integer() and y.is_integer()) :
            return 0
        else : 
            px = int(np.floor(x))
            py = int(np.floor(y))

            relCoord = complex(x - px, y - py)
            u, v = relCoord.real, relCoord.imag

            grads = [self.grads[self._gradIdx(px + p[0], py + p[1])] for p in self._lattice()]
            noises = [complexDot(g, relCoord - complex(*p)) for g, p in zip(grads, self._lattice())]

            nx0 = noises[0] * (1 - self._f(u)) + noises[2] * self._f(u)
            nx1 = noises[1] * (1 - self._f(u)) + noises[3] * self._f(u)

            nxy = nx0 * (1 - self._f(v)) + nx1 * self._f(v)
            return nxy

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

def treeKCut (tree, k) :
    """
    Given a tree, make k cuts. Ideally the k cuts 
    should give evenly sized sets.

    Parameters
    ----------
    tree : Tree
        Hierachical clustering from which
        k evenly sized sets.
    """
    def selector (a, b) : 
        if a['level'] < b['level'] : 
            return a
        elif a['level'] > b['level'] :
            return b
        elif len(a['ids']) > len(b['ids']) :
            return a
        else :
            return b

    def split (T, best) :
        level, arr = None, None
        if len(best['ids']) > 1 :
            arr = copy.deepcopy(best['ids'])
            level = best['level']
        else :
            i = best['ids'][0]
            arr = list(T.neighbors(i))
            level = best['level'] + 1
        index = random.randint(0, len(arr) - 1)
        left = { 'level':level, 'ids':arr[:index] + arr[index+1:]}
        right = { 'level': level, 'ids':[arr[index]] }
        return left, right

    T = tree.tree
    r = findRoot(T)
    candidates = [{'level': 0, 'ids': [r]}]
    leaves = [];
    while len(candidates) + len(leaves) < k :
        best = reduce(selector, candidates)
        candidates.remove(best)
        left, right = split (T, best)
        if len(left['ids']) > 1 or T.out_degree(left['ids'][0]) > 0 :
            candidates.append(left)
        else :
            leaves.append(left)

        if len(right['ids']) > 1 or T.out_degree(right['ids'][0]) > 0 :
            candidates.append(right)
        else :
            leaves.append(right)

    candidates.extend(leaves)
    cuts = list(map(lambda c : list(
            more_itertools.flatten(
                map(lambda i : T.nodes[i]['pathSet'], c['ids'])))
            , candidates))
    return cuts

def hierarchicalClusterCompareFM (t1, t2) : 
    """
    Implementation of: 

        A Method for Comparing Two Hierarchical Clusterings

    This method gives statistics using which it can be
    decided whether two hierarchical clusterings are 
    similar.

    FM stands for Fowlkes and Mallows.

    Parameters
    ----------
    t1 : Tree
        Tree one.
    t2 : Tree
        Tree two.
    """
    assert t1.nPaths == t2.nPaths
    n = t1.nPaths
    bs = []
    es = [] 
    for k in range(2, 5): 
        cuts1 = treeKCut(t1, k)
        cuts2 = treeKCut(t2, k)
        M = np.zeros((k, k))
        for i, ci in enumerate(cuts1) : 
            for j, cj in enumerate(cuts2) :
                M[i, j] = len(set(ci) & set(cj))
        tk = (M * M).sum() - n
        mi, mj = M.sum(axis=0), M.sum(axis=1)
        pk = (mi ** 2).sum() - n
        qk = (mj ** 2).sum() - n
        bk = tk / np.sqrt(pk * qk)
        ek = np.sqrt(pk * qk) / (n * (n - 1))
        bs.append(bk)
        es.append(ek)
    return np.array(bs)

def getCubicMatrix () : 
    """
    Helper method to get matrix 
    for parametric representation
    of cubic beziers.
    """
    M = np.array([[1 , 0 , 0 , 0], 
                  [-3, 3 , 0 , 0], 
                  [3 , -6, 3 , 0], 
                  [-1, 3 , -3, 1]])
    t = np.linspace(0, 1, 4)
    T = np.vstack((t**0, t**1, t**2, t**3)).T
    return T @ M

def getQuadraticMatrix () : 
    """
    Helper method to get matrix 
    for parametric representation
    of quadratic beziers.
    """
    M = np.array([[1 , 0 , 0 ], 
                  [-2, 2 , 0 ], 
                  [1 , -2, 1 ]])
    t = np.linspace(0, 1, 3)
    T = np.vstack((t**0, t**1, t**2)).T
    return T @ M

def fitCubicBezierTo4Points (P) :
    """
    Fit a cubic bezier to 4 points,
    the minimum required for a 
    non-degenerate cubic bezier to exist.

    The 4 points' parameter t is assumed
    to be equispaced in the interval [0, 1].

    Parameters
    ----------
    P : np.ndarray
        4 points as complex numbers 
        with the real part denoting the 
        x-coordinate and imaginary part 
        denoting the y-coordinate.

    Returns
    -------
    svg.CubicBezier
    """
    C = np.linalg.pinv(getCubicMatrix()) @ P
    return svg.CubicBezier(*C)

def fitQuadraticBezierTo3Points (P) : 
    """
    Fit a quadratic bezier to 3 points,
    the minimum required for a 
    non-degenerate quadratic bezier to exist.

    The 3 points' parameter t is assumed
    to be equispaced in the interval [0, 1].

    Parameters
    ----------
    P : np.ndarray
        3 points as complex numbers 
        with the real part denoting the 
        x-coordinate and imaginary part 
        denoting the y-coordinate.

    Returns
    -------
    svg.QuadraticBezier
    """
    C = np.linalg.pinv(getQuadraticMatrix()) @ P
    return svg.QuadraticBezier(*C)

def fitPathToPoints (P) : 
    """
    Fit a series of splines (cubic beziers
    or quadratic beziers or simply lines)
    to a set of points.

    Also, in case the curve is closed, 
    the first point in P should be the 
    same as the last point.

    Parameters
    ----------
    P : np.ndarray
        A list of points
    """
    n = P.size
    if n == 2 : 
        return svg.Path([Line(*P)])
    elif n == 3 : 
        return svg.Path([fitQuadraticBezierTo3Points(P)])
    else : 
        segments = [fitCubicBezierTo4Points(P[:4])]
        for seg in more_itertools.chunked(P[4:], 3) : 
            prevSeg = segments[-1]
            s = prevSeg.end
            control = 2 * s - prevSeg.control2

            if len(seg) == 3 : 
                P_ = np.array([[s, *seg]]).T
                A = getCubicMatrix()
                C = np.array([0, 1, 0, 0])
                A_ = np.hstack((np.vstack((2 * A.T @ A, C)), np.vstack((C.reshape((-1, 1)), 0))))
                b = np.vstack([2 * A.T @ P_, control])
                controls = np.linalg.pinv(A_) @ b
                controls = np.ravel(controls.T)[:-1]
                segments.append(svg.CubicBezier(*controls))
            elif len(seg) == 2 : 
                _, e = seg
                segments.append(svg.QuadraticBezier(s, control, e))
            else :
                segments.append(svg.Line(s, seg[0]))

    return svg.Path(*segments)

def removeOneOutDegreeNodesFromTree (tree) : 
    """
    In many SVGs, it is the case that 
    there are unecessary groups which contain
    one a single group.

    In general these don't capture the hierarchy
    because the one out-degree nodes can be 
    removed without altering the grouping.

    Hence we have this function which removes
    one out-degree nodes from a networkx tree.

    Example
    -------
    >>> tree = nx.DiGraph()
    >>> tree.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (4, 6)])
    >>> tree = removeOneOutDegreeNodesFromTree(tree)
    >>> print(tree.number_of_nodes())

    Since the nodes 0 and 2 have out-degree 1,
    they'll be deleted and we'll be left
    with 5 nodes.
   
    Parameters
    ----------
    tree : nx.DiGraph
    """
    def remove (n) : 
        children = list(tree.neighbors(n))
        if tree.out_degree(n) == 1 : 
            child = children[0]
            tree.remove_node(n)
            return remove(child)
        else : 
            newEdges = list(itertools.product([n], map(remove, children)))
            tree.add_edges_from(newEdges)
            return n

    remove(findRoot(tree))
    topOrder = list(nx.topological_sort(tree))
    relabelDict = dict(zip(topOrder, range(tree.number_of_nodes())))
    tree = nx.relabel_nodes(tree, relabelDict)
    return tree

def levenshteinDistance (a, b, matchFn, costFn) : 
    """
    Calculate the optimal tree edit distance
    between two lists, given the costs of
    matching two items on the list and 
    the costs of deleting an item in the
    optimal match.

    Example
    -------
    >>> skipSpace = lambda x : 0 if x == ' ' else 1
    >>> match = lambda a, b : 0 if a == b else 1
    >>> print(levenshteinDistance('sumitchaturvedi', 'sumi t ch at urvedi', match, skipSpace))

    In the above example, since we don't penalize for 
    deleting spaces, the levenshtein distance is 
    going to be 0.

    Parameters
    ----------
    a : list
        First string.
    b : list
        Second string.
    matchFn : lambda
        The cost of matching the
        ith character in a with the 
        jth character in b.
    costFn : lambda 
        The penalty of skipping over
        one character from one of the
        strings.
    """
    n = len(a)
    m = len(b)
    row = list(range(n + 1))

    for j in range(1, m + 1) : 
        newRow = [0 for _ in range(n + 1)]
        for i in range(n + 1): 
            if i == 0 : 
                newRow[i] = j
            else : 
                match = matchFn(a[i-1], b[j-1])
                cost1 = costFn(a[i-1])
                cost2 = costFn(b[j-1])
                newRow[i] = min(row[i-1] + match, newRow[i-1] + cost1, row[i] + cost2)
        row = newRow
    return row[n]

def svgTreeEditDistance (t1, t2, paths, vbox) :
    """
    Compute two Tree objects (which capture)
    SVG document structure. Each node in the 
    tree is a group of paths. We have a 
    similarity function between these nodes 
    based on the levenshtein distance between the 
    path strings.
    
    Parameters
    ----------
    t1 : Data.Tree
        Tree one.
    t2 : Data.Tree
        Tree two.
    paths : list
        List of paths.
    vbox : list
        Bounding Box.
    """

    def pathMatchFn (a, b) : 

        def curveMatchFn (a, b) :
            """
            If the curve parameters match
            within a distance threshold, then
            we let them be.
            """
            if isinstance(a, type(b)) : 
                if isinstance(a, svg.Line) :
                    cond1 = abs(a.start - b.start) < thresh
                    cond2 = abs(a.end   - b.end  ) < thresh
                    return 0 if cond1 and cond2 else 1
                elif isinstance(a, svg.QuadraticBezier) : 
                    cond1 = abs(a.start   - b.start  ) < thresh
                    cond2 = abs(a.end     - b.end    ) < thresh
                    cond3 = abs(a.control - b.control) < thresh
                    return 0 if cond1 and cond2 and cond3 else 1
                elif isinstance(a, svg.CubicBezier) : 
                    cond1 = abs(a.start    - b.start   ) < thresh
                    cond2 = abs(a.end      - b.end     ) < thresh
                    cond3 = abs(a.control1 - b.control1) < thresh
                    cond4 = abs(a.control2 - b.control2) < thresh
                    return 0 if cond1 and cond2 and cond3 and cond4 else 1
                elif isinstance(a, svg.Arc) : 
                    cond1 = abs(a.start  - b.start ) < thresh
                    cond2 = abs(a.end    - b.end   ) < thresh
                    cond3 = abs(a.radius - b.radius) < thresh
                    return 0 if cond1 and cond2 and cond3 else 1
                else :
                    raise TypeError
            else :
                return 1

        def curveDelFn (a) : 
            return 1 
    
        if (a, b) not in cachedMatchVals : 
            pathA = paths[a].path
            pathB = paths[b].path
            maxLen = max(len(pathA), len(pathB))

            pathMatch = levenshteinDistance(pathA, pathB, curveMatchFn, curveDelFn)
            normalized = pathMatch / maxLen

            cachedMatchVals[(a, b)] = normalized
            cachedMatchVals[(b, a)] = normalized
        
        return cachedMatchVals[(a, b)]

    def pathDelFn (a) : 
        return 1

    def cost (u1, u2) :
        
        def pathSetDist (x, y) : 
            ps1 = t1.tree.nodes[x]['pathSet']
            ps2 = t2.tree.nodes[y]['pathSet']
            lev = levenshteinDistance(ps1, ps2, pathMatchFn, pathDelFn)
            maxLen = max(len(ps1), len(ps2))
            return lev / maxLen

        nbrs1 = list(t1.tree.neighbors(u1))
        nbrs2 = list(t2.tree.neighbors(u2))
        degree1 = t1.tree.out_degree(u1)
        degree2 = t2.tree.out_degree(u2)
        if degree1 == 0 and degree2 == 0 :
            return pathSetDist(u1, u2)
        elif degree1 == 0 : 
            return sub2[u2] - sub1[u1];
        elif degree2 == 0 :
            return sub1[u1] - sub2[u2]
        else :
            prod = list(itertools.product(nbrs1, nbrs2))
            costs = list(map(lambda x : cost(*x) + pathSetDist(*x), prod))
            nbrs2 = [str(_) for _ in nbrs2]
            costdict = dict(zip(prod, costs))
            return bestAssignmentCost(costdict)

    
    xmin, ymin, xmax, ymax = vbox
    diagonal = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
    thresh = diagonal / 10

    cachedMatchVals = dict() 
    sub1, sub2 = dict(), dict()
    subtreeSize(t1.root, t1.tree, sub1)
    subtreeSize(t2.root, t2.tree, sub2)
    c = cost(t1.root, t2.root)
    return c

def leaves (tree) :
    """
    Returns the leaf nodes in a
    directed tree.

    Parameters
    ----------
    tree : nx.DiGraph
    """
    return list(filter (lambda x : tree.out_degree(x) == 0, tree.nodes))

def nonLeaves (tree) :
    """
    Returns the internal nodes in a
    directed tree.

    Parameters
    ----------
    tree : nx.DiGraph
    """
    return list(filter (lambda x : tree.out_degree(x) > 0, tree.nodes))

def pairwiseDisjoint (setList) :
    """
    Checks whether the sets in the
    list are pairwise disjoint.

    Parameters
    ----------
    setList : list
    """
    for s1, s2 in itertools.combinations(setList, 2) :
        if not s1.isdisjoint(s2) : 
            return False
    return True

def hasDuplicates (lst) :
    """
    Check whether a list has duplicates.

    Parameters
    ----------
    lst : list
    """
    return len(set(lst)) != len(lst)

def mergeTrees (trees) : 
    """
    Merge a list of trees into a single 
    tree. 
    
    The catch is that the leaf nodes
    which represent path indices in our setup
    have to be distinctly labeled across the
    trees. 
    
    So we only relabel the internal
    nodes so that while composing, we
    don't introduce edges which shouldn't
    be there. 

    Finally we add a new root node with
    these subtrees as the children.

    Example
    -------
    >>> tree1 = nx.DiGraph()
    >>> tree2 = nx.DiGraph()
    >>> tree1.add_edges_from([(3, 1), (3, 2)])
    >>> tree2.add_edges_from([(4, 3), (5, 4)])
    >>> print(mergeTrees([tree1, tree2]).edges)

    Parameters
    ----------
    trees : list
    """
    def relabelTree (tree) : 
        nonlocal maxIdx
        internalNodes = nonLeaves(tree)

        newId = range(maxIdx, maxIdx + len(internalNodes))
        newLabels = dict(zip(internalNodes, newId))

        maxIdx += len(internalNodes)
        return nx.relabel_nodes(tree, newLabels, copy=True)

    allLeaves = list(more_itertools.collapse(map(leaves, trees)))

    maxIdx = max(allLeaves) + 1

    relabeledTrees = list(map(relabelTree, trees))
    roots = map(findRoot, relabeledTrees)

    newTree = nx.compose_all(relabeledTrees)
    newEdges = list(map(lambda x : (maxIdx, x), roots))
    newTree.add_edges_from(newEdges)
    newTree.nodes[maxIdx]['pathSet'] = allLeaves

    return newTree

def removeIndices (lst, indices) :
    """
    In place removal of items at given indices. 
    Obtained from : 

    https://stackoverflow.com/questions/497426/deleting-multiple-elements-from-a-list

    Parameters
    ----------
    lst : list
        List from which indices are to be
        removed.
    indices : list
        List of indices. No error checking is 
        done.
    """
    for i in sorted(indices, reverse=True):
        del lst[i] 

def argmax(lst) :
    """
    Compute the argmax of a list.

    Parameters
    ----------
    lst : list
    """
    return next(filter(lambda x : max(lst) == lst[x], range(len(lst))))

def argmax(lst) :
    """
    Compute the argmax of a list.

    Parameters
    ----------
    lst : list
    """
    return next(filter(lambda x : max(lst) == lst[x], range(len(lst))))

def argmin(lst) :
    """
    Compute the argmin of a list.

    Parameters
    ----------
    lst : list
    """
    return next(filter(lambda x : min(lst) == lst[x], range(len(lst))))

def zipDirs (dirList) :
    """ 
    A common operation is to get SVGs
    or graphs from different directories
    and match them up and perform 
    some operations on them. 

    Since the files have similar names,
    we sort them before zipping them.

    For example, dir1 may have E269.svg
    while dir2 may have E269.json. Sorting 
    will ensure that while zipping, these 
    two files are together.

    Parameters
    ----------
    dirList : list
        List of directories to zip.
    """
    filesInDirList = list(map(os.listdir, dirList))
    for i in range(len(dirList)) :
        filesInDirList[i].sort()
        filesInDirList[i] = [os.path.join(dirList[i], f) for f in filesInDirList[i]]
    return zip(*filesInDirList)


def subgraph (G, predicate=lambda x : x['weight'] < 1e-3) :
    """ 
    Get a subgraph of G by
    removing edges which don't 
    satisfy a given predicate.

    Useful when filtering those edges
    from the symmetry graph which have
    high error.

    Parameters
    ----------
    G : nx.Graph()
        Normal Graph
    predicate : lambda 
        Predicate over edges to be kept
        in the subgraph
    """
    G_ = copy.deepcopy(G)
    badEdges = list(itertools.filterfalse(lambda e : predicate(G_.edges[e]), G_.edges))
    G_.remove_edges_from(badEdges)
    return G_

def optimalBipartiteMatching (costTable) :
    """
    Return the minimum cost bipartite 
    matching.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the 
        cost of matching that pair
        together.
    """
    G = nx.Graph()

    for key, val in costTable.items() :
        i, j = key
        G.add_node(i, bipartite=0)
        G.add_node(str(j), bipartite=1)
        G.add_edge(i, str(j), weight=val)

    matching = bipartite.minimum_weight_full_matching(G)
    return matching

def bestAssignmentCost (costTable) :
    """
    Compute the minimum total
    cost assignment.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the 
        cost of matching that pair
        together.
    """
    G = nx.Graph()

    for key, val in costTable.items() :
        i, j = key
        G.add_node(i, bipartite=0)
        G.add_node(str(j), bipartite=1)
        G.add_edge(i, str(j), weight=val)

    matching = bipartite.minimum_weight_full_matching(G)
    # Each edge gets counted twice
    cost = sum([G.edges[e]['weight'] for e in matching.items()]) / 2
    return cost

def subtreeSize(s, t, subSize) :
    """
    Calculate the size of each
    subtree in the rooted tree t.

    Parameters
    ----------
    s : int
        The current vertex
    t : nx.DiGraph()
        The tree
    subSize : dict
        Store the size of the subtree
        rooted at each vertex.
    """
    subSize[s] = 1
    nbrs = list(t.neighbors(s))
    if t.out_degree(s) != 0 : 
        for u in nbrs :
            subtreeSize(u, t, subSize)
            subSize[s] += subSize[u]

def match (rt1, rt2) :
    """
    Given two full rooted binary trees,
    find the minimum nodes needed to be 
    added to either of the trees to make
    them isomorphic.

    >>> t1 = nx.DiGraph()
    >>> t2 = nx.DiGraph()
    >>> t1.add_edges_from([(0,1), (1,3), (0,2)])
    >>> t2.add_edges_from([(0,1), (0,2), (2,3)])
    >>> match((0, t1), (0, t2))

    Parameters
    ----------
    rt1 : tuple
        A (root index, nx.DiGraph) tuple representing
        the first rooted tree.
    rt2 : tuple
        The second rooted tree.
    """
    r1, t1 = rt1
    r2, t2 = rt2
    sub1 = dict()
    sub2 = dict()
    subsetsize(r1, t1, sub1)
    subsetsize(r2, t2, sub2)

    def cost (u1, u2) :
        nbrs1 = list(t1.neighbors(u1))
        nbrs2 = list(t2.neighbors(u2))
        degree1 = len(nbrs1)
        degree2 = len(nbrs2)
        if degree1 == 0 and degree2 == 0 :
            return 0
        elif degree1 == 0 : 
            return sub2[u2] - sub1[u1];
        elif degree2 == 0 :
            return sub1[u1] - sub2[u2]
        else :
            prod = list(itertools.product(nbrs1, nbrs2))
            costs = list(map(lambda x : cost(*x), prod))
            nbrs2 = [str(_) for _ in nbrs2]
            costdict = dict(zip(prod, costs))
            return bestassignmentcost(costdict)

    return cost(r1, r2)

def parallelize(indirList, outdir, function, writer, **kwargs) :
    """ 
    Use multiprocessing library to 
    parallelize preprocessing. 

    A very common operation that I
    have to do is to apply some
    operations on some files in a
    directory and write to some other
    directory. Often, these operations 
    are very slow. This function takes
    advantage of the multiple cpus to 
    distribute the computation.

    Parameters
    ----------
    indirList : list
        paths to the input directories
    outdir : str
        path to the output directory
    function : lamdba 
        function to be applied on 
        each file in the input directory
    writer : lambda
        function which writes the results
        of function application to a file of
        given name
    """

    def subTask (inChunk, outChunk) :
        for i, o in zip(inChunk, outChunk) :
            obj = function(i, **kwargs)
            writer(obj, o, **kwargs)

    cpus = mp.cpu_count()

    names = list(map(lambda x : os.path.splitext(x)[0], os.listdir(indirList[0])))
    names.sort()

    inFullPaths = list(zipDirs(indirList))
    outFullPaths = [os.path.join(outdir, name) for name in names]

    inChunks = more_itertools.divide(cpus, inFullPaths)
    outChunks = more_itertools.divide(cpus, outFullPaths)

    processList = []

    for inChunk, outChunk in zip(inChunks, outChunks) :
        p = mp.Process(target=subTask, args=(inChunk, outChunk,))
        p.start()
        processList.append(p)

    for p in processList :
        p.join()

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
    if makeSquare : 
        subprocess.call(['inkscape', f'-h {H}', f'-w {W}', '-z', '-f', svgFile, '-j', '-e', outFile])
    else : 
        subprocess.call(['inkscape', '-z', '-f', svgFile, '-j', '-e', outFile])

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
    ps = [paths[i][0] for i in lst]
    attrs = [paths[i][1].attrib for i in lst]
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

class GraphReadWrite () :
    """
    Convenience class to handle graph
    read/write operations so that
    I don't have to keep searching 
    for networkx documentation.

    Two graph types are supported right now:
        1) 'cytoscape'
        2) 'tree'
    """

    def __init__ (self, graphType) :
        """
        Parameters
        ----------
        graphType : str
            Indicate which graph type
            to read/write.
        """
        self.graphType = graphType

    def read (self, inFile) :
        """
        Parameters
        ----------
        inFile : str
            Path to input file to write to.
        """
        with open(inFile, 'r') as fd :
            dct = json.loads(fd.read())
        if self.graphType == 'cytoscape' :
            return nx.readwrite.json_graph.cytoscape_graph(dct)
        elif self.graphType == 'tree' :
            return nx.readwrite.json_graph.tree_graph(dct)
        
        raise ValueError ('Unsupported Graph Type')

    def write (self, G, outFile) :
        """
        Parameters
        ----------
        G : nx.Graph / (nx.DiGraph, int)
            If graphType is cytoscape then
            it is a nx.Graph, else, it is the 
            tree along with the root index.
        outFile : str
            Path to the output file.
        """
        dct = None
        if self.graphType == 'cytoscape' :
            dct = nx.readwrite.json_graph.cytoscape_data(G)
        elif self.graphType == 'tree' :
            T, r = G
            dct = nx.readwrite.json_graph.tree_data(T, r)

        if dct is None :
            raise ValueError ('Unsupported Graph Type')

        with open(outFile, 'w+', encoding='utf-8') as fd :
            json.dump(dct, fd, ensure_ascii=False, indent=4)

def optimalRotationAndTranslation (pts1, pts2) : 
    """
    Get the optimal rotation matrix
    and translation vector to transform
    pts1 to pts2 in linear least squares
    sense.

    The output is the rotation matrix,
    the translation vector and the error.

    Parameters
    ----------
    pts1 : np.ndarray
        Points in the first set
    pts2 : np.ndarray
        Points in the second set
    """
    centroid1 = np.mean(pts1, axis=0)
    centroid2 = np.mean(pts2, axis=0)

    pts1_ = pts1 - centroid1
    pts2_ = pts2 - centroid2

    H = pts1_.T @ pts2_

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
            R[:, 1] *= -1

    t = -(R @ centroid1.T) + centroid2.T
    return R, t, np.linalg.norm(pts2 - (pts1 @ R.T + t))

def isometryWithAlignment(path1, 
                          path2, 
                          ts1=np.arange(0, 1, 0.05),
                          ts2=np.arange(0, 1, 0.05)) :
    """
    Given two paths and an alignment, find
    the best isometric transformation which will
    take path1 to path2. 

    By definition of isometric transformation,
    we consider reflection, rotation and 
    translation.

    Return whether reflection was needed,
    the rotation matrix, the translation
    vector and the error.

    Parameters
    ----------
    path1 : svg.Path
        The first path.
    path2 : svg.Path
        The second path.
    ts1 : np.ndarray
        The sequence of curve parametrization
        points for path1.
    ts2 : np.ndarray
        The sequence of curve parametrization
        points for path2. The correspondence
        between points on each path
        uses these two sequences.
    """

    reflection = np.array([[1, 0], [0, -1]]) 
        
    pts1 = []
    pts2 = []

    for t1, t2 in zip(ts1, ts2) : 
        pt1 = path1.point(t1)
        pt2 = path2.point(t2)

        pts1.append([pt1.real, pt1.imag])
        pts2.append([pt2.real, pt2.imag])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    refPts1 = pts1 @ reflection.T

    R1, t1, e1 = optimalRotationAndTranslation(pts1, pts2)
    R2, t2, e2 = optimalRotationAndTranslation(refPts1, pts2)

    if e1 < e2 : 
        return False, R1, t1, e1
    else :
        return True, R2, t2, e2 


def isometry(path1, path2) :
    """
    Find the best isometry to transform
    path1 into path2. 

    If the two paths are actually the
    same, then there are multiple 
    possible ways of matching them up. 
    
    Fortunately, because we know the 
    curve parametrizations, there are only
    a few ways of matching them up. 

    We loop over all possible matchings
    and find the transformation which
    minimizes the error.

    >>> paths = svg.Document('file.svg').flatten_all_paths()
    >>> ref, R, t, e = isometry(paths[0][0], paths[1][0])
    >>> print("Reflection: ", ref)
    >>> print("Rotation: ", R)
    >>> print("Translation: ", t)
    >>> print("Error: ", e)

    Parameters
    ----------
    path1 : svg.Path
        The first path.
    path2 : svg.Path
        The second path.
    """
    l1 = path1.length()
    l2 = path2.length()
    
    if path1.isclosed() and path2.isclosed() : 
        r, R, t, err = None, None, None, None
        
        rng = np.arange(0, 1, 0.05)
        for i in range(len(rng))  :
            fst = rng[i:]
            snd = rng[:i]
            ts = np.concatenate((fst, snd))

            r_, R_, t_, err_ = isometryWithAlignment(path1, path2, ts1=ts)

            if not err :
                r, R, t, err = r_, R_, t_, err_
            elif err_ < err :
                r, R, t, err = r_, R_, t_, err_

        return r, R, t, err
    else :
        return isometryWithAlignment(path1, path2)
        
def complexLE (x, y) :
    """
    Lesser than equal to for complex numbers.

    Parameters
    ----------
    x : complex
    y : complex
    """
    if x.real < y.real :
        return True
    elif x.real > y.real :
        return False
    else :
        return x.imag < y.imag

def complexDot (x, y) :
    """
    Dot product between complex
    numbers treating them as vectors
    in R^2.

    Parameters
    ----------
    x : complex
    y : complex
    """
    return (x.real * y.real) + (x.imag * y.imag)

def complexCross (x, y) : 
    """
    Cross product between complex
    numbers treating them as vectors
    in R^2.

    Parameters
    ----------
    x : complex
    y : complex
    """
    v1 = np.array([x.real, x.imag, 0.0]) 
    v2 = np.array([y.real, y.imag, 0.0]) 
    return np.linalg.norm(np.cross(v1,v2))

    
def d2 (path, docbb, bins=10, nSamples=100) :
    """
    Compute the d2 descriptors of the path.
    Take two random points on the curve
    and make an histogram of the distance 
    between them. We use this or fd in our
    experiments.

    Parameters
    ----------
    path : svg.Path
        Input path.
    docbb : list
        bounding box of the document.
    bins : int
        Number of histogram bins. Also the
        dimension of the descriptor.
    nSamples : int
        Number of samples while making
        histogram.
    """
    xmin, xmax, ymin, ymax = path.bbox()
    rmax = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)

    if rmax <= 1e-10 : 
        return np.ones(bins)

    hist = np.zeros(bins)
    binSz = rmax / bins

    L = path.length() 

    for i in range(nSamples) : 

        pt1 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt2 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        
        r = abs(pt1 - pt2) 

        bIdx = int(r / binSz)

        if bIdx == bins :
            bIdx -= 1

        hist[bIdx] += 1

    hist = hist / hist.sum()
    return hist.tolist()

def shell(path, docbb, bins=10) :
    """
    Like d2 but instead of random points,
    we use points at fixed intervals on
    the curve.

    Parameters
    ----------
    path : svg.Path
        Input path.
    docbb : list
        bounding box of the document.
    bins : int
        Number of histogram bins. Also the
        dimension of the descriptor.
    nSamples : int
        Number of samples while making
        histogram.
    """
    xmin, xmax, ymin, ymax = path.bbox()
    rmax = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)
    K = 100
    L = path.length()

    samples = []
    for i in range(K) : 
        s = (i / K) * L
        samples.append(path.point(path.ilength(s, s_tol=1e-6)))
    
    d2 = [abs(x-y) for x in samples for y in samples if complexLE(x, y)]
    hist = np.zeros(bins)
    binSz = rmax / bins

    if binSz == 0 : 
        return np.ones(bins)

    for dist in d2 :
        b = int(dist/binSz)
        if b >= bins : 
            b = bins - 1
        hist[b] += 1
    return hist

def a3 (path, docbb, bins=10, nSamples=100) :
    """
    Compute the a3 descriptors of the path.
    Take three random points on the curve
    and make an histogram of the angle 
    subtended.

    Parameters
    ----------
    path : svg.Path
        Input path.
    docbb : list
        bounding box of the document.
    bins : int
        Number of histogram bins. Also the
        dimension of the descriptor.
    nSamples : int
        Number of samples while making
        histogram.
    """
    hist = np.zeros(bins)
    binSz = np.pi / bins
    L = path.length()
    i = 0
    while i < nSamples :

        pt1 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt2 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt3 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        
        v1 = pt1 - pt3
        v2 = pt2 - pt3
        
        if v1 == 0 or v2 == 0 : 
            continue
        
        cos = complexDot(v1, v2) / (abs(v1) * abs(v2))
        cos = np.clip(cos, -1, 1)

        theta = np.arccos(cos)

        bIdx = int(theta / binSz)

        if bIdx == bins : 
            bIdx -= 1

        hist[bIdx] += 1

        i += 1

    return hist

def d3 (path, docbb, bins=10, nSamples=100) :
    """
    Compute the d3 descriptors of the path.
    Take three random points on the curve
    and make their area histogram.

    Parameters
    ----------
    path : svg.Path
        Input path.
    docbb : list
        bounding box of the document.
    bins : int
        Number of histogram bins. Also the
        dimension of the descriptor.
    nSamples : int
        Number of samples while making
        histogram.
    """
    xmin, xmax, ymin, ymax = path.bbox()
    area = (ymax - ymin) * (xmax - xmin) 

    if area <= 1e-10 : 
        return np.ones(bins)

    hist = np.zeros(bins)
    binSz = area / bins

    L = path.length() 

    for i in range(nSamples) : 

        pt1 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt2 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt3 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        
        v1 = pt1 - pt3
        v2 = pt2 - pt3

        trArea = complexCross(v1, v2) / 2

        bIdx = int(trArea / binSz)

        if bIdx == bins :
            bIdx -= 1

        hist[bIdx] += 1

    return hist 

def fd (path, docbb, nSamples=100, freqs=10) :
    """
    Compute the fourier descriptors of the
    path with respect to its centroid.

    Parameters
    ----------
    path : svg.Path
        Input path. 
    docbb : list
        Bounding Box of the document.
    nSamples : int
        Sampling frequency for the path
    """
    ts = np.arange(0, 1, 1 / nSamples)
    L = path.length()
    if L == 0 :
        return np.ones(min(nSamples, 20))
    pts = np.array([path.point(path.ilength(t * L, s_tol=1e-5)) for t in ts])
    pts = pts - pts.mean()
    an = np.fft.fft(pts)
    pos = an[1:nSamples//2]
    neg = an[nSamples//2 + 1:]
    pos = pos[:freqs]
    neg = neg[-freqs:]
    newAn = np.hstack([an[0], pos, neg])
    reals = np.array(list(map(lambda x : x.real, newAn)))
    imags = np.array(list(map(lambda x : x.imag, newAn)))
    newAn = np.hstack([reals, imags])
    return newAn

def relbb (path, docbb) :
    """ 
    Compute the relative bounding box
    of the path with respect to the 
    document's bounding box.

    Parameters
    ----------
    path : svg.Path
        Input path.
    docbb : list
        The svg document's bounding box.
    """
    xmin, xmax, ymin, ymax = path.bbox()
    x1 = (xmin - docbb[0]) / (docbb[2] - docbb[0])
    x2 = (xmax - docbb[0]) / (docbb[2] - docbb[0])

    y1 = (ymin - docbb[1]) / (docbb[3] - docbb[1])
    y2 = (ymax - docbb[1]) / (docbb[3] - docbb[1])

    return [x1, x2, y1, y2]

def symGraph (paths, **kwargs) : 
    """
    Return a complete graph over the 
    paths. The edge attributes between 
    the vertices are the optimal
    transformations along with the 
    associated errors.

    Parameters
    ----------
    paths : list
        List of paths over which 
        symmetry has to be checked.
    """
    paths = list(enumerate(paths))
    
    G = nx.Graph() 
    
    for p1, p2 in itertools.product(paths, paths) :
        i, j = p1[0], p2[0]
        pi, pj = p1[1][0], p2[1][0]
    
        if i < j :
            r, R, t, err = isometry(pi, pj)
            G.add_edge(
                i, j, 
                reflection=r, 
                rotation=R.tolist(), 
                translation=t.tolist(),
                error=err,
                etype='symmetry'
            )

    return subgraph(G, lambda x : x['error'] < 1)

def adjGraph (paths, **kwargs) :
    """
    Return a graph over the 
    paths. We have coarse estimates
    of adjacency between two paths.
    
    We simply check whether their 
    bounding boxes intersect or not.

    Parameters
    ----------
    paths : list
        List of paths over which 
        adjacency has to be checked.
    """
    paths = list(enumerate(paths))
    G = nx.Graph()
    
    for i in range(len(paths)) :
        G.add_node(i)

    for p1, p2 in itertools.product(paths, paths) : 
        i, j = p1[0], p2[0]
        pi, pj = p1[1][0], p2[1][0]

        if i != j and pi != pj :
            if pi.approx_adj(pj) :
                G.add_edge(i, j, etype='adjacency')

    return G

def areaGraph (paths, **kwargs) : 
    """
    Return a complete graph over the 
    paths. Where the edge weights are the 
    areas of intersection between pairs
    of paths.
    
    Parameters
    ----------
    paths : list
        List of paths over which 
        adjacency has to be checked.
    """
    paths = list(enumerate(paths))
    G = nx.Graph()
    
    for i in range(len(paths)) :
        G.add_node(i)

    for p1, p2 in itertools.product(paths, paths) : 
        i, j = p1[0], p2[0]
        pi, pj = p1[1][0], p2[1][0]

        if i != j and pi != pj :
            weight = pathIntersectionArea(p1[1], p2[1], kwargs['vbox'])
            G.add_edge(i, j, etype='area', weight=weight)

    return G

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
    rs = randomString(10)
    rasterize(svgFilePath, f'/tmp/{rs}.png', H=H, W=W)
    img = image.imread(f'/tmp/{rs}.png')
    return img[:, :, :3]

def pathIntersectionArea (path1, path2, vbox) : 
    """
    Calculate the area of intersection of two
    paths normalized by the document view box.

    This is done by first rasterizing and then
    taking the intersection of the bitmaps. 

    Parameters
    ----------
    path1 : svg.FlattenedPath
        Path 1
    path2 : svg.FlattenedPath
        Path 2
    vbox : list
        Viewbox of the document
    """
    # Fills with black hopefully
    fillDict = {'fill' : [0, 0, 0]}
    path1_ = copy.deepcopy(path1)
    path2_ = copy.deepcopy(path2)
    path1_[1].attrib = fillDict
    path2_[1].attrib = fillDict

    singlePathSvg(path1_, vbox, '/tmp/o1.svg')
    singlePathSvg(path2_, vbox, '/tmp/o2.svg')

    rasterize('/tmp/o1.svg', '/tmp/o1.png', makeSquare=False)
    rasterize('/tmp/o2.svg', '/tmp/o2.png', makeSquare=False)

    img1 = image.imread('/tmp/o1.png')
    img2 = image.imread('/tmp/o2.png')

    img1 = img1[:, :, 3]
    img2 = img2[:, :, 3]

    intersect = img1 * img2
    n, m = intersect.shape
    fraction = intersect.sum() / (n * m)
    return fraction

def graphFromSvg (svgFile, graphFunction) : 
    """
    Produce a relationship graph for 
    the svg file

    Parameters
    ----------
    svgFile : str
        path to svg file.
    graphFunction : lambda 
        path relation computer.
    """
    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    G = graphFunction(paths, vbox=doc.get_viewbox())
    return G

def relationshipGraph (svgFile, relFunctions) :
    """
    Take the svg, find graphs
    for all relationship types
    and combine them into one single
    graph for clustering.

    Parameters
    ----------
    svgFile : str
        path to svg file.
    relFunctions : list
        List of graph computing functions
    """
    graphs = map(lambda x : graphFromSvg(svgFile, x), relFunctions)
    graph = None
    for g in graphs :
        if graph is None:
            graph = g
        else :
            graph = nx.compose(graph, g)
    return graph

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
#        imagebox = OffsetImage(img, zoom=0.2)
        imagebox = OffsetImage(img, zoom=0.08)
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

def hik (a, b) :
    """
    Histogram Intersection Kernel.

    A distance measure between two
    histograms.

    Parameters
    ----------
    a : np.ndarray
        The first histogram.
    b : np.ndarray
        The second histogram.
    """
    return np.sum(np.minimum(a, b))

def chi2 (a, b) :
    """
    Chi-squared distance between
    two vectors.

    Parameters
    ----------
    a : np.ndarray
        The first vector.
    b : np.ndarray
        The second vector.
    """
    n = (a - b) * (a - b) 
    d = a + b
    n[d > 0] /= d[d > 0]
    return np.sum(n)

def bhattacharya (a, b) :
    """
    Bhattacharya distance between
    two vectors.

    Parameters
    ----------
    a : np.ndarray
        The first vector.
    b : np.ndarray
        The second vector.
    """
    return 1 - np.sum(np.sqrt(a * b))

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

def graphCluster (G, algo, doc) :
    """
    Hierarchical clustering of a graph into
    a dendogram. Given an algorithm to 
    partition a graph into two sets, the
    this class produces a tree by recursively 
    partitioning the graphs induced on these
    subsets till each subset contains only a
    single node.

    Parameters
    ----------
    G : nx.Graph
        The path relationship graph
    algo : lambda
        The partitioning algorithm
        to be used
    doc : svg.Document
        Used to set node attributes.
    """

    def cluster (lst) :
        """
        Recursively cluster the
        subgraph induced by the 
        vertices present in the 
        list.

        lst : list
            List of vertices to
            be considered.
        """
        nonlocal idx
        tree.add_node(idx)
        curId = idx
        idx += 1
        if len(lst) != 1 :
            subgraph = G.subgraph(lst)
            l, r = algo(subgraph)
            lId = cluster(list(l))
            rId = cluster(list(r))
            tree.add_edge(curId, lId)
            tree.add_edge(curId, rId)
            # Add indices of paths in this subtree.
            tree.nodes[curId]['pathSet'] = lst
        else :
            pathId = lst.pop()
            tree.nodes[curId]['pathSet'] = [pathId]

        return curId

    paths = doc.flatten_all_paths()
    vbox = doc.get_viewbox()
    tree = nx.DiGraph()
    idx = 0
    root = 0
    cluster(list(G.nodes))
    return tree, root

def treeApplyRootFirst (T, r, function) :
    """
    Apply function to all nodes in the
    tree.

    Parameters
    ----------
    T : nx.DiGraph
        The tree.
    r : object
        Root of the tree.
    function : lambda
        A function which takes as
        input the tree, current node 
        and performs some operation.
    """
    function(T, r, T.neighbors(r))
    for child in T.neighbors(r) :
        treeApplyRootFirst(T, child, function)

def treeApplyChildrenFirst (T, r, function) : 
    """
    Apply function to all nodes in the
    tree.

    Parameters
    ----------
    T : nx.DiGraph
        The tree.
    r : object
        Root of the tree.
    function : lambda
        A function which takes as
        input the tree, current node 
        and performs some operation.
    """
    for child in T.neighbors(r) :
        treeApplyChildrenFirst(T, child, function)
    function(T, r, T.neighbors(r))

def treeMap (T, r, function) : 
    """
    Apply a function on each node and accumulate
    results in a list.

    Parameters
    ----------
    T : nx.DiGraph
        The tree.
    r : object
        Root of the tree.
    function : lambda
        A function which takes as
        input the tree, current node 
        and performs some operation.
    """
    results = []
    for child in T.neighbors(r) :
        results.extend(treeMap(T, child, function))
    results.append(function(T, r, T.neighbors(r)))
    return results

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

def svgStringToBitmap (svgString) :
    svgName = randomString(10) + '.svg'
    svgName = osp.join('/tmp', svgName)

    pngName = randomString(10) + '.png'
    pngName = osp.join('/tmp', pngName)

    with open(svgName, 'w+') as fd :
        fd.write(svgString)

    rasterize(svgName, pngName)
    img = image.imread(pngName)

    os.remove(svgName)
    os.remove(pngName)

    return img.tolist()

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

def findRoot (tree) :
    """
    Find the root of a tree.

    Parameters
    ----------
    tree : nx.DiGraph
        Rooted tree with unknown root.
    """
    return next(nx.topological_sort(tree))

def configReadme (path) :
    """
    I think it is a good idea to save the 
    current snapshot of the config 
    file along with the trained model.

    Parameters
    ----------
    path : str
        Path to save to.
    """
    with open('Config.py', 'r') as fd: 
        content = fd.read()
    with open(path, 'w+') as fd :
        fd.write('Configuration used:\n')
        fd.write('```\n')
        fd.write(content)
        fd.write('```\n')

def getTreeStructureFromSVG (svgFile) : 
    """
    Infer the tree structure from the
    XML document. 

    Parameters
    ----------
    svgFile : str
        Path to the svgFile.
    """

    def buildTreeGraph (element) :
        """
        Recursively create networkx tree and 
        add path indices to the tree.

        Parameters
        ----------
        element : Element
            Element at this level of the
            tree.
        """
        nonlocal r
        curId = r
        r += 1
        T.add_node(curId)
        if element.tag in childTags : 
            zIdx = allNodes.index(element)
            pathIdx = zIndexMap[zIdx]
            T.nodes[curId]['pathSet'] = [pathIdx]
        else : 
            childIdx = []
            validTags = lambda x : x.tag in childTags or x.tag == groupTag
            children = list(map(buildTreeGraph, filter(validTags, element)))
            T.add_edges_from(list(itertools.product([curId], children)))
            pathSet = list(more_itertools.collapse([T.nodes[c]['pathSet'] for c in children]))
            T.nodes[curId]['pathSet'] = pathSet
        return curId

    childTags = [
        '{http://www.w3.org/2000/svg}rect',
        '{http://www.w3.org/2000/svg}circle',
        '{http://www.w3.org/2000/svg}ellipse',
        '{http://www.w3.org/2000/svg}line', 
        '{http://www.w3.org/2000/svg}polyline',
        '{http://www.w3.org/2000/svg}polygon',
        '{http://www.w3.org/2000/svg}path',
    ]
    groupTag = '{http://www.w3.org/2000/svg}g'

    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    zIndexMap = dict([(p.zIndex, i) for i, p in enumerate(paths)])
    tree = doc.tree
    root = tree.getroot()
    allNodes = list(root.iter())
    T = nx.DiGraph()
    r = 0
    buildTreeGraph (root)
    T = removeOneOutDegreeNodesFromTree(T)
    return Data.Tree(T)

def listdir (path) :
    """
    Convenience function to get 
    full path details while calling os.listdir

    Parameters
    ----------
    path : str
        Path to be listed.
    """
    paths = [osp.join(path, f) for f in os.listdir(path)]
    paths.sort()
    return paths

class AllPathDescriptorFunction () : 
    """ 
    Given a descriptor function, convert it into
    one that acts on all paths
    """

    def __init__ (self, descFunction) : 
        self.descFunction = descFunction

    def __call__ (self, paths, vbox) : 
        descs = []
        for path in paths : 
            descs.append(self.descFunction(path, vbox))
        return np.vstack(descs)

def mpiiPoseDataSet () :
    """
    Extract pose annotations from the dataset
    Based on: 

        1) https://tensorlayercn.readthedocs.io/zh/latest/_modules/tensorlayer/files/dataset_loaders/mpii_dataset.html
        2) http://human-pose.mpi-inf.mpg.de/#download
    """
    annotations = []
    mat = sio.loadmat("./mpii_human_pose_v1_u12_1.mat")
    for anno in mat['RELEASE']['annolist'][0, 0][0] : 
        annotations.append([])
        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(annopoints, head_x1s, head_y1s, head_x2s,
                                                                     head_y2s):
                if annopoint.size:
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[int(_j_id)] = [float(_x), float(_y)]
                    if len(joint_pos) == 16 : 
                        joint_pos = list(joint_pos.items())
                        joint_pos.sort()
                        joint_pos = [p for _, p in joint_pos]
                        annotations[-1].append(joint_pos)
    return annotations

if __name__ == "__main__" : 
    poses = list(more_itertools.flatten(mpiiPoseDataSet()))
    for i, sample in enumerate(pose): 
        p = Pose(sample)
        p.getDocument().save(f'./Examples/body{i}.svg')
