from Dataset import SVGDataSet
from treeOps import * 
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import skew
from scipy import stats

def areaStd (data) : 
    std = []
    for t in data : 
        for u in t.nodes : 
            if t.out_degree(u) == 0 : 
                continue
            boxes = [t.nodes[v]['bbox'] for v in t.neighbors(u)]
            area = [(b[2] + b[3])/2 for b in boxes]
            std.append(np.std(area))
    plt.hist(std, bins=20)
    plt.show()

def centerY (data) : 
    center = []
    for t in data : 
        for b in t.pathViewBoxes : 
            center.append(b[1] + (b[3]/2))
    plt.hist(center, bins=20)
    plt.show()

def centerXStd (data) : 
    std = []
    for t in data : 
        for u in t.nodes : 
            if t.out_degree(u) == 0 : 
                continue
            boxes = [t.nodes[v]['bbox'] for v in t.neighbors(u)]
            centerX = [b[0] + (b[2] / 2) for b in boxes]
            std.append(np.std(centerX))
    plt.hist(std, bins=20)
    plt.show()

def centerYStd (data) : 
    std = []
    for t in data : 
        for u in t.nodes : 
            if t.out_degree(u) == 0 : 
                continue
            boxes = [t.nodes[v]['bbox'] for v in t.neighbors(u)]
            centerX = [b[1] + (b[3] / 2) for b in boxes]
            std.append(np.std(centerX))
    plt.hist(std, bins=20)
    plt.show()

def main () :
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load all the data
    trainDir = commonConfig['train_directory']
    trainData = SVGDataSet(trainDir, 'adjGraph', 10, useColor=False)
    thing = np.concatenate([t.lcaMatrix.flatten() for t in trainData])
    print(skew(thing))
    print(skew(np.log(1 + thing)))
    print(skew(stats.boxcox(thing + 1e-3)[0]))
    plt.hist(thing)
    plt.show()
    plt.hist(np.log(1 + thing))
    plt.show()
    plt.hist(stats.boxcox(thing + 1e-3)[0], bins=2)
    plt.show()

if __name__ == "__main__" : 
    main()
