""" suggero - https://mi-lab.org/files/2013/10/suggero.pdf """
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from vectorrvnn.geometry import *
from vectorrvnn.utils.svg import cachedPaths
from vectorrvnn.utils.graph import hac2nxDiGraph
import networkx as nx

USE_LITE = True

SUGGERO_ADVANCED = [localProximity, globalProximity, 
        fourierDescriptorDistance, fillDistance, 
        strokeDistance, strokeWidthDifference, 
        endpointDistance, parallelismDistance]
SUGGERO_LITE = SUGGERO_ADVANCED[:-2]

def affinityMatrix (doc, affinityFn) :
    G = relationshipGraph(doc, affinityFn, True)
    M = nx.to_numpy_matrix(G, weight=affinityFn.__name__)
    M = M / (M.max() + 1e-5)
    return M

def combinedAffinityMatrix (doc, affinityFns, weights) : 
    affinityMatrices = [affinityMatrix(doc, fn) for fn in affinityFns]
    combinedMatrix = sum(w * M for w, M in zip(weights, affinityMatrices))
    return combinedMatrix

def suggero (doc) : 
    paths = cachedPaths(doc)
    subtrees = list(range(len(paths)))
    paths = [paths[i] for i in subtrees]
    affinityFns = SUGGERO_LITE if USE_LITE else SUGGERO_ADVANCED
    nfns = len(affinityFns)
    uniformWts = (1 / nfns) * np.ones(nfns)
    M = combinedAffinityMatrix(doc, affinityFns, uniformWts)
    M = M[:, subtrees][subtrees, :]
    agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
    agg.fit(M)
    return hac2nxDiGraph(subtrees, agg.children_)

#def processDir(DIR) : 
#    try : 
#        OUTDIR = 'unsupervised_v2'
#        _, NAME = osp.split(DIR)
#        SVGFILE = osp.join(DIR, f'{NAME}.svg')
#        if osp.getsize(SVGFILE) < 5e4 and 5 <= len(svg.Document(SVGFILE).flatten_all_paths()) <= 150 : 
#            datapt = osp.join(OUTDIR, str(NAME))
#            os.mkdir(datapt)
#            with open(osp.join(datapt, 'file.txt'), 'w+') as fp : 
#                fp.write(SVGFILE)
#            inferredTree = suggero(SVGFILE)
#            nx.write_gpickle(inferredTree, osp.join(datapt, 'tree.pkl'))
#    except Exception : 
#        pass

# if __name__ == "__main__" : 
#     DATASET = '/net/voxel07/misc/extra/data/sumitc/datasetv1'
#     OUTDIR = 'unsupervised_v2'
#     files = listdir(DATASET)
#     with mp.Pool(mp.cpu_count()) as p : 
#         list(tqdm(p.imap(processDir, files, chunksize=30), total=len(files)))
