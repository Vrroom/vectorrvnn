from graphIO import GraphReadWrite
import networkx as nx
import os 
import svgpathtools as svg
import os.path as osp
from pg import DB
import json
import more_itertools
from svgIO import setSVGAttributes
from treeOps import findRoot, treeApplyChildrenFirst
from vis import matplotlibFigureSaver, treeImageFromGraph
from raster import singlePathSvg, rasterize

def nodeLinks2tree (nodeLinks) : 
    def aggregatePathSets (T, r, neighbors) :
        if T.out_degree(r) > 0 : 
            childrenSets = map(lambda x : T.nodes[x]['pathSet'], neighbors)
            T.nodes[r]['pathSet'] = list(more_itertools.flatten(childrenSets))
    nodes = nodeLinks['nodes']
    T = nx.DiGraph()
    for node in nodes : 
        if 'parent' in node : 
            id = node['id']
            pId = node['parent']
            T.add_edge(pId, id)
    for node in nodes : 
        id = node['id']
        if len(node['children']) == 0 : 
            T.nodes[id]['pathSet'] = [id]
    treeApplyChildrenFirst(T, findRoot(T), aggregatePathSets)
    return T

if __name__ == "__main__" : 
    DATABASE_URL = 'postgres://plgakpajwcwmue:a47dcde86cd9e7cf6f88a818560a4e495e7a36a4afc7995d2ab97cc3e79ec0e7@ec2-34-198-243-120.compute-1.amazonaws.com:5432/dbemgkh7ofctf9'
    db = DB(DATABASE_URL)
    query = 'select vectorgraphs.id, vectorgraphs.graph, vectorimages.svg from vectorimages inner join vectorgraphs on vectorimages.id = vectorgraphs.id;'
    items = db.query(query).dictresult()
    dataDir = '/net/voxel07/misc/me/sumitc/vectorrvnn/ManuallyAnnotatedDataset'
    import pdb
    pdb.set_trace()
    for i, item in enumerate(items) : 
        graph = item['graph']
        svgString = item['svg']
        tree = nodeLinks2tree(graph)
        assert nx.is_tree(tree)
        doc = svg.Document(None)
        doc.fromString(svgString)
        doc.normalize_viewbox()
        paths = doc.flatten_all_paths()
        vb = doc.get_viewbox()
        # setSVGAttributes(tree, paths, vb)
        # matplotlibFigureSaver(treeImageFromGraph(tree), f'./Viz/{i}')
        datumPath = osp.join(dataDir, str(i))
        os.mkdir(datumPath)
        doc.save(osp.join(datumPath, f'{i}.svg'))
        GraphReadWrite('tree').write((tree, findRoot(tree)), osp.join(datumPath, f'{i}.json'))
