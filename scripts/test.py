from train import *
import pandas as pd
import numpy as np
from vectorrvnn.utils import *
from collections import defaultdict

def iou (a, b) : 
    return len(set(a).intersection(set(b))) / len(set(a).union(set(b)))

def loadModelFromCkpt (opts) : 
    """ 
    opts should have minimum of 
    checkpoints_dir, load_ckpt and device 
    """ 
    ckptpath = osp.join(opts.checkpoints_dir, opts.load_ckpt) 
    ModelCls = opts.modelcls
    model = ModelCls(opts).float()
    state_dict = torch.load(ckptpath, map_location=opts.device)
    model.load_state_dict(state_dict['model'])
    model.to(opts.device)
    model.eval()
    print(model)
    return model

def scores2df (ts1, ts2, methodName) : 
    ctedscore = avg(map(norm_cted, ts1, ts2))
    fmi1score = avg(map(partial(fmi, level=1), ts1, ts2))
    fmi2score = avg(map(partial(fmi, level=2), ts1, ts2))
    fmi3score = avg(map(partial(fmi, level=3), ts1, ts2))
    return pd.DataFrame(
        data=[[ctedscore, fmi1score, fmi2score, fmi3score]],
        index=[methodName],
        columns=['cted', 'fmi1', 'fmi2', 'fmi3']
    )

def nodeoverlap2df (ts_gt, ts_inf, methodName) : 
    scores = [] 
    for T, T_ in zip(ts_gt, ts_inf) : 
        gNodes = nonLeaves(T)
        gNodes.remove(findRoot(T))
        for n in gNodes : 
            ps = T.nodes[n]['pathSet']
            maxIoU = max([iou(ps, T_.nodes[_]['pathSet']) for _ in T_.nodes])
            scores.append(maxIoU)
    score = np.mean(scores)
    return pd.DataFrame(
        data=[[score]], 
        index=[methodName], 
        columns=['maxiou']
    ) 

def test (opts) : 
    _, _, _, _, testData = buildData(opts)
    model = loadModelFromCkpt(opts)
    # evaluate scores on test set
    exprDir = osp.join(opts.checkpoints_dir, opts.name)
    logFile = osp.join(exprDir, f'test_eval.csv')
    # evaluate greedy tree
    testData = [_ for _ in testData if _.nPaths < opts.max_len] 
    dd_tree = list(map(model.greedyTree, testData))
    dd = scores2df(testData, dd_tree, "Ours-DD")
    # evaluate containment guided
    cg_tree = list(map(model.containmentGuidedTree, testData))
    cg = scores2df(testData, cg_tree, "Ours-CG")
    # combine them into a single dataframe for saving
    combined = pd.concat([dd, cg])
    combined.to_csv(logFile)
    # now do the node overlap test
    logFile = osp.join(exprDir, f'node_overlap.csv') 
    publicDomain = '../data/PublicDomainVectors'
    publicDomain = osp.join(osp.dirname(osp.realpath(__file__)), publicDomain)
    svgFiles = [f for f in allfiles(publicDomain) if f.endswith('svg')][:500] 
    publicdata  = [SVGData(_) for _ in svgFiles]
    # Filter out graphics with too many paths. 
    publicdata = [_ for _ in publicdata if _.nPaths < opts.max_len] 
    # evaluate greedy tree
    dd_tree = list(map(model.greedyTree, publicdata))
    dd = nodeoverlap2df(publicdata, dd_tree, 'Ours-DD') 
    # evaluate containment guided
    cg_tree = list(map(model.containmentGuidedTree, publicdata)) 
    cg = nodeoverlap2df(publicdata, cg_tree, 'Ours-CG') 
    combined = pd.concat([dd, cg])
    combined.to_csv(logFile)

if __name__ == "__main__" :  
    opts = Options().parse()
    setSeed(opts)
    # inject model class 
    opts = dict(opts._asdict())
    opts['modelcls'] = NN_CLASSES[opts['modelcls']]
    Option = namedtuple('Option', [k for k in opts]) 
    opts = Option(*[v for _, v in opts.items()]) 
    test(opts)
    
     
