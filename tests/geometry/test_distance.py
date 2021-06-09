from vectorrvnn.geometry.distance import *
from vectorrvnn.utils import *
import os
import os.path as osp
import svgpathtools as svg
from sklearn.cluster import AgglomerativeClustering

def fpAssert (v1, v2) : 
    assert(abs(v1 - v2) < 1e-4)

def test_proximity () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    circleDoc = svg.Document(osp.join(chdir, 'data', 'circle.svg'))
    lp = localProximity(circleDoc, 0, 1)
    gp = globalProximity(circleDoc, 0, 1)
    fpAssert(lp, 40 * np.sqrt(2))
    fpAssert(gp, 40 * np.sqrt(2))

def test_pathattrs () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = ['line.svg', 'circle.svg', 'path.svg']
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    for f in svgs : 
        doc = svg.Document(f)
        nPaths = len(doc.paths()) 
        for i in range(nPaths) : 
            for j in range(i + 1, nPaths) : 
                assert(strokeDistance(doc, i, j) < 1e-4)
                assert(fillDistance(doc, i, j) < 1e-4)
                assert(strokeWidthDifference(doc, i, j) < 1e-4)

def test_endpoint () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = ['endpoint_1.svg', 'endpoint_2.svg', 'endpoint_3.svg']
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    docs = [svg.Document(f) for f in svgs]
    assert(endpointDistance(docs[0], 0, 1) < 0.5)
    assert(endpointDistance(docs[1], 0, 1) > 0.7)
    assert(endpointDistance(docs[2], 0, 1) > 0.7)

def test_parallel () :
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = ['parallel_1.svg', 'parallel_2.svg', 
            'parallel_3.svg', 'parallel_4.svg']
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    docs = [svg.Document(f) for f in svgs]
    assert(parallelismDistance(docs[0], 0, 1) < 0.1)
    assert(parallelismDistance(docs[1], 0, 1) < 0.1)
    assert(parallelismDistance(docs[2], 0, 1) < 0.1)
    assert(parallelismDistance(docs[3], 0, 1) > 0.3)

def test_isometry () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = ['isometry_1.svg', 'isometry_2.svg', 'isometry_3.svg']
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    docs = [svg.Document(f) for f in svgs]
    assert(isometricDistance(docs[0], 0, 1) < 0.01)
    assert(isometricDistance(docs[1], 0, 1) < 0.01)
    assert(isometricDistance(docs[2], 0, 1) > 0.1)

def test_areaintersection () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    fname = osp.join(chdir, 'data', 'intersection.svg')
    doc = svg.Document(fname)
    a1 = areaIntersectionDistance(doc, 0, 1)
    a2 = areaIntersectionDistance(doc, 1, 2)
    a3 = areaIntersectionDistance(doc, 2, 0)
    assert(a1 == 0) 
    assert(a3 > a2)

def test_autogroup_stroke_similarity () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = [
        '1F452.svg',
        '20182-Home-loan-interest-rate-icon-vector-clip-art.svg',
        'autogroup_stroke.svg'
    ]
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    docs = [svg.Document(f) for f in svgs]
    fnName = 'autogroupStrokeSimilarity'
    graphs = [relationshipGraph(doc, autogroupStrokeSimilarity, True) 
            for doc in docs]
    matrices = [nx.to_numpy_matrix(g, weight=fnName) 
            for g in graphs]
    matrices = [1 - m for m in matrices]
    # all vals should be bounded in range
    assert(all([0 <= m.min() <= m.max() <= 1 for m in matrices]))
    # all matrices should be symmetric 
    assert(all([np.linalg.norm(m.T - m) < 1e-3 
        for m in matrices]))
    # Rasterize the groups after clustering to visualize
    for i, (m, doc) in enumerate(zip(matrices, docs)) : 
        agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
        agg.fit(m)
        T = hac2nxDiGraph(list(range(m.shape[0])), agg.children_)
        T.doc = doc
        figure = treeImageFromGraph(T) 
        matplotlibFigureSaver(figure, 
            osp.join(chdir, 'out', f'autogroup-stroke-{i}'))
    T_ = getTreeStructureFromSVG(svgs[-1])
    distance = ted(T, T_) / (T.number_of_nodes() + T_.number_of_nodes())
    assert(distance < 0.1)

def test_autogroup_color_similarity () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = [
        '1F452.svg',
        '20182-Home-loan-interest-rate-icon-vector-clip-art.svg',
        'autogroup_stroke.svg'
    ]
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    docs = [svg.Document(f) for f in svgs]
    fnName = 'autogroupColorSimilarity'
    containmentGraphs = [relationshipGraph(doc, bboxContains, False) 
            for doc in docs]
    containmentGraphs = [subgraph(cg, lambda x : x['bboxContains']) 
            for cg in containmentGraphs]
    graphs = [
        relationshipGraph(
            doc, autogroupColorSimilarity, 
            True, containmentGraph=cg) 
        for doc, cg in zip(docs, containmentGraphs)
    ]
    matrices = [nx.to_numpy_matrix(g, weight=fnName) 
            for g in graphs]
    matrices = [1 - m for m in matrices]
    # all vals should be bounded in range
    assert(all([0 <= m.min() <= m.max() <= 1 for m in matrices]))
    # all matrices should be symmetric 
    assert(all([np.linalg.norm(m.T - m) < 1e-3 
        for m in matrices]))
    # Rasterize the groups after clustering to visualize
    for i, (m, doc) in enumerate(zip(matrices, docs)) : 
        agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
        agg.fit(m)
        T = hac2nxDiGraph(list(range(m.shape[0])), agg.children_)
        T.doc = doc
        figure = treeImageFromGraph(T) 
        matplotlibFigureSaver(figure, 
            osp.join(chdir, 'out', f'autogroup-color-{i}'))

def test_autogroup_shape_similarity () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgFile = 'shape.svg'
    fullpath = osp.join(chdir, 'data', svgFile)
    doc = svg.Document(fullpath)
    fnName = 'autogroupShapeHistogramSimilarity'
    g = relationshipGraph(doc, 
            autogroupShapeHistogramSimilarity, True) 
    m = nx.to_numpy_matrix(g, weight=fnName)
    m = 1 - m
    assert(0 <= m.min() <= m.max() <= 1)
    assert(np.linalg.norm(m.T - m) < 1e-3)
    # Rasterize the groups after clustering to visualize
    agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
    agg.fit(m)
    T = hac2nxDiGraph(list(range(m.shape[0])), agg.children_)
    T.doc = doc
    figure = treeImageFromGraph(T) 
    matplotlibFigureSaver(figure, 
        osp.join(chdir, 'out', f'autogroup-shape'))
    T_ = getTreeStructureFromSVG(fullpath)
    distance = ted(T, T_) / (T.number_of_nodes() + T_.number_of_nodes())
    assert(distance < 0.15)

def test_autogroup_area_similarity () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgFile = 'area.svg'
    fullpath = osp.join(chdir, 'data', svgFile)
    doc = svg.Document(fullpath)
    fnName = 'autogroupAreaSimilarity'
    g = relationshipGraph(doc, 
            autogroupAreaSimilarity, True) 
    m = nx.to_numpy_matrix(g, weight=fnName)
    m = 1 - m
    assert(0 <= m.min() <= m.max() <= 1)
    assert(np.linalg.norm(m.T - m) < 1e-3)
    # Rasterize the groups after clustering to visualize
    agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
    agg.fit(m)
    T = hac2nxDiGraph(list(range(m.shape[0])), agg.children_)
    T_ = getTreeStructureFromSVG(fullpath)
    T.doc = doc
    figure = treeImageFromGraph(T) 
    matplotlibFigureSaver(figure, 
        osp.join(chdir, 'out', f'autogroup-area'))
    distance = ted(T, T_) / (T.number_of_nodes() + T_.number_of_nodes())
    assert(distance < 0.15)

def test_autogroup_space_similarity () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgFile = 'space.svg'
    fullpath = osp.join(chdir, 'data', svgFile)
    doc = svg.Document(fullpath)
    fnName = 'autogroupPlacementDistance'
    g = relationshipGraph(doc, 
            autogroupPlacementDistance, True) 
    m = nx.to_numpy_matrix(g, weight=fnName)
    m = m / m.max()
    assert(0 <= m.min() <= m.max() <= 1)
    assert(np.linalg.norm(m.T - m) < 1e-3)
    # Rasterize the groups after clustering to visualize
    agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
    agg.fit(m)
    T = hac2nxDiGraph(list(range(m.shape[0])), agg.children_)
    T_ = getTreeStructureFromSVG(fullpath)
    T.doc = doc
    figure = treeImageFromGraph(T) 
    matplotlibFigureSaver(figure, 
        osp.join(chdir, 'out', f'autogroup-place'))
    distance = ted(T, T_) / (T.number_of_nodes() + T_.number_of_nodes())
    assert(distance < 0.16)
