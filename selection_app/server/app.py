import os
import os.path as osp
from datetime import datetime
import uuid
from flask import Flask, jsonify, request, session, make_response
from flask_session import Session
import pickle
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.network import *
from vectorrvnn.interfaces import *
from vectorrvnn.baselines import *
from vectorrvnn.trainutils import *
from functools import lru_cache, partial
import sys
from strokeAnalyses import suggest
import svgpathtools as svg

def rootdir():  
    return osp.abspath(osp.dirname(__file__))

# Setup application.
app = Flask(__name__, static_url_path='', static_folder='../client/build')
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

# Setup model and data.
opts = Options().parse([
    '--backbone', 'resnet18',
    '--checkpoints_dir', osp.join(rootdir(), '../../results'),
    '--dataroot', osp.join(rootdir(), '../../data/All'),
    '--embedding_size', '64',
    '--hidden_size', '128', '128',
    '--load_ckpt', 'expt3/training_end.pth',
    '--loss', 'cosineSimilarity',
    '--modelcls', 'ThreeBranch',
    '--name', 'server',
    '--phase', 'test',
    '--rasterize_thread_local', 'True',
    '--sim_criteria', 'negativeCosineSimilarity',
    '--temperature', '0.1',
    '--use_layer_norm', 'True',
    '--seed', '2'
])
setSeed(opts)
_, _, _, _, data = buildData(opts)
model = buildModel(opts)

# Available Backends
backends = ['triplet'] #, 'suggero', 'autogroup']
tools = ['slider', 'scribble', 'toggle']

treeInference = {
    'triplet': model.greedyTree,
    'suggero': suggero,
    'autogroup': partial(autogroup, opts=opts)
}

@lru_cache(maxsize=1024)
def baseId2Pickle (baseId, backend):
    file = osp.join(rootdir(), f'/../data/{backend}', f'{baseId}.pkl')
    with open(file, 'rb') as fd :
        data = pickle.load(fd)
    return data

@app.route('/')
def root():  
    session['id'] = uuid.uuid4()
    session['tool'] = tool = rng.choice(tools)
    session['backend'] = backend = rng.choice(backends)
    with open(f'{app.static_folder}/index.html') as fp :
        content = fp.read()
    resp = make_response(content)
    resp.set_cookie('tool', tool)
    resp.set_cookie('backend', backend)
    return resp

@app.route('/task', methods=['POST', 'GET']) 
def task () : 
    backend = session.get('backend')
    T = data[0]
    with open(T.svgFile) as fp : 
        svg = fp.read()
    forest = nxGraph2appGraph(T)
    return jsonify(target=[0, 1], svg=svg, id=0, forest=forest)

@app.route('/example', methods=['POST', 'GET']) 
def example() :
    id = rng.randint(0, len(data) - 1)
    T = data[id]
    with open(T.svgFile) as fp : 
        svg = fp.read()
    backend = session.get('backend')
    print(backend)
    forest = nxGraph2appGraph(treeInference[backend](T))
    return jsonify(id=id, svg=svg, forest=forest)

@app.route('/stroke', methods=['POST', 'GET']) 
def stroke () :
    backend = session.get('backend')
    id = request.json['id']
    stroke = request.json['stroke']
    radius = request.json['radius']
    t = data[int(id)]
    inference = suggest(t, stroke, treeInference[backend], radius)
    return jsonify(suggestions=inference)

@app.route('/surveyquestion', methods=['POST', 'GET'])
def surveyquestion () :
    id = session.get('id')
    tool = session.get('tool')
    backend = session.get('backend')
    question = request.json['question']
    score = request.json['score']
    surveyFile = osp.join(rootdir(), '../data/survey', str(id) + '.csv')
    currentTime = str(datetime.now())
    with open(surveyFile, 'a') as fd : 
        fd.write(f'{id}, {tool}, {backend}, {question}, {score}, {currentTime}\n')
    return 'ok'

@app.route('/logcomments', methods=['POST', 'GET'])
def logcomments () : 
    id = session.get('id')
    tool = session.get('tool')
    backend = session.get('backend')
    currentTime = str(datetime.now())
    comments = request.json['comments']
    commentsFile = osp.join(rootdir(), '../data/comments', str(id) + '.csv')
    with open(commentsFile, 'a') as fd : 
        fd.write(f'{id}, {tool}, {backend}, {comments}, {currentTime}\n')
    return 'ok'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
