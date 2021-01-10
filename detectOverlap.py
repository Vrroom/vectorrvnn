from fuzzywuzzy import fuzz
import numpy as np
import svgpathtools as svg
from itertools import combinations
from itertools import starmap
from osTools import listdir
from tqdm import tqdm
from raster import * 
import matplotlib.pyplot as plt
from functools import partial
from more_itertools import unzip
import random
from subprocess import call
from scipy.spatial.distance import directed_hausdorff

def iou (box1, box2) : 
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xm = max(x1, x2)
    xM = min(x1 + w1, x2 + w2) 
    ym = max(y1, y2)
    yM = min(y1 + h1, y2 + h2) 
    intersection = max(0, xM - xm) * max(0, yM - ym)
    union = (w1 * h1) + (w2 * h2) - intersection + 1e-3
    return intersection / union

def hausdorff (a, b) : 
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

def icp (a, b) : 
    aName = '/tmp/' + randomString(10) + '.csv'
    bName = '/tmp/' + randomString(10) + '.csv'
    writePtCloud(a, aName)
    writePtCloud(b, bName)
    outName = '/tmp/' + randomString(10) + '.txt'
    with open(outName, 'w+') as out : 
        call([icp_exec, aName, bName], stdout=out)
    T = np.loadtxt(outName)
    return T

def path2PtCloud (path, docViewBox) : 
    pts = []
    for i in np.linspace(0, 1, 100) : 
        point = path.point(i) 
        x = point.real / docViewBox[2]
        y = point.imag / docViewBox[3]
        pts.append([x, y])
    return np.array(pts)

def writePtCloud (ptCloud, csv) : 
    with open(csv, 'w+') as fd : 
        for row in ptCloud : 
            x, y = row
            fd.write(f'{x}, {y}\n')

def checkOverlap (path1, path2, doc) : 
    if fuzz.ratio(path1.path.d(), path2.path.d()) > 0.95 : 
        return True
    else : 
        docViewbox = doc.get_viewbox()
        ptCloud1 = path2PtCloud(path1.path, docViewbox)
        ptCloud2 = path2PtCloud(path2.path, docViewbox)
        if hausdorff(ptCloud1, ptCloud2) < 0.02 : 
            return True
        elif iou(path1.path.bbox(), path2.path.bbox()) < 0.5 : 
            return False 
        else : 
            T = icp(ptCloud1, ptCloud2) 
            return np.linalg.norm(T - np.eye(3)) < 0.02
