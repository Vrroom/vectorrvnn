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
    db = DB('localVectors')
    query = 'select vectorgraphs.id, vectorgraphs.graph, vectorimages.svg from vectorimages inner join vectorgraphs on vectorimages.id = vectorgraphs.id;'
    items = db.query(query).dictresult()
    dataDir = '/Users/amaltaas/BTP/vectorrvnn/ManuallyAnnotatedDataset'
    for i, item in enumerate(items) : 
        try : 
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

            for j, path in enumerate(paths): 
                rasterFilePath = osp.join(datumPath, f'path_{j+1}.png')
                singlePathSvg(path, vb, '/tmp/o.svg')
                rasterize('/tmp/o.svg', rasterFilePath, H=28, W=28)

            GraphReadWrite('tree').write((tree, findRoot(tree)), osp.join(datumPath, f'{i}.json'))
        except Exception as e:
            print(f'{e} happened')
