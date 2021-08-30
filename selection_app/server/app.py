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
from vectorrvnn.interface import *
from vectorrvnn.baselines import *
from vectorrvnn.trainutils import *
from functools import lru_cache
import sys
from strokeAnalyses import suggest
import svgpathtools as svg

# Setup application.
app = Flask(__name__, static_url_path='', static_folder='../client/build')
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

# Setup model.
opts = Options().parse([
    '--backbone', 'resnet18',
    '--sim_criteria', 'negativeCosineSimilarity',
    '--modelcls', 'OneBranch',
    '--phase', 'test',
    '--load_ckpt', osp.join(rootdir(), 
        '/../../results/onebranch/best_0-780-08-20-2021-10-33-29.pth')
])
model = buildModel(opts)

# Available Backends
backends = ['triplet', 'suggero', 'autogroup']
tools = ['slider', 'scribble']

treeInference = {
    'triplet': model.greedyTree,
    'suggero': suggero,
    'autogroup': autogroup
}

def rootdir():  
    return osp.abspath(osp.dirname(__file__))

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

@app.route('/taskgraphic', methods=['POST', 'GET']) 
def taskgraphic () : 
    backend = session.get('backend')
    taskList = listdir(osp.join(rootdir(), f'../data/{backend}Trees'))
    pkl = rng.choice(taskList)
    with open(pkl, 'rb') as fd : 
        forest = pickle.load(fd)
    id = getBaseName(pkl)
    svg = forest.svg
    forest = nxGraph2appGraph(forest)
    im = f'./target/{id}.png'
    return jsonify(im=im, svg=svg, id=id, forest=forest)

@app.route('/db', methods=['POST', 'GET']) 
def db() :
    backend = session.get('backend')
    randomFile = rng.choice(listdir(osp.join(rootdir(), 'svgs')))
    id = getBaseName(randomFile)
    with open(randomFile) as fp : 
        svg = fp.read()
    t = nx.DiGraph()
    nPaths = len(Document(randomFile).flatten_all_paths())
    t.add_nodes_from(range(nPaths))
    forest = nxGraph2appGraph(t)
    return jsonify(svg=svg, id=id, forest=forest)

@app.route('/demofile') 
def demofile () :
    file = '103.pkl'
    id = getBaseName(file)
    forest = baseId2Pickle(id, 'triplet')
    svg = forest.svg
    forest = nxGraph2appGraph(forest)
    return jsonify(svg=svg, id=id, forest=forest)

@app.route('/demostroke', methods=['POST', 'GET'])
def demostroke () :
    file = '103.pkl'
    id = getBaseName(file)
    t = baseId2Pickle(id, 'triplet')
    userStroke = request.json['stroke']
    radius = request.json['radius']
    inference = suggest(t, userStroke, treeInference['triplet'], radius)
    return jsonify(suggestions=inference)

@app.route('/stroke', methods=['POST', 'GET']) 
def stroke () :
    backend = session.get('backend')
    pkls = listdir(osp.join(rootdir(), f'../data/{backend}Trees'))
    id = request.json['id']
    pkl = [p for p in pkls if getBaseName(p) == id].pop()
    with open(pkl, 'rb') as fd : 
        t = pickle.load(fd)
    userStroke = request.json['stroke']
    radius = request.json['radius']
    inference = suggest(t, userStroke, treeInference[backend], radius)
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

@app.route('/clicklog', methods=['POST', 'GET'])
def clicklog () :
    id = session.get('id')
    currentTime = str(datetime.now())
    surveyFile = osp.join(rootdir(), '../data/clicklog', str(id) + '.csv')
    with open(surveyFile, 'a') as fd :
        fd.write(f'{id}, {currentTime}, {request.json}\n')
    return 'ok'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
