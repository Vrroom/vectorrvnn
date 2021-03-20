import pickle
import os
import os.path as osp
from TripletSVGData import *
import matplotlib.pyplot as plt
from osTools import listdir
from tqdm import tqdm
from raster import SVGSubset2NumpyImage2
import svgpathtools as svg

#DIR = '/misc/extra/data/sumitc/suggeroPicklesIndividual'
#
#for f in tqdm(listdir(DIR)) :
#    with open(f, 'rb') as fp: 
#        t = pickle.load(fp)
#    pngName = './pngs/' + osp.splitext(osp.split(t.svgFile)[1])[0] + '.png'
#    plt.imsave(pngName, t.bigImage[150:450, 150:450], format='png')

for i, f in tqdm(enumerate(listdir('./svgs'))) :
    doc = svg.Document(f)
    paths = doc.flatten_all_paths()
    im = SVGSubset2NumpyImage2(doc, list(range(len(paths))), 300, 300, alpha=True)
    plt.imsave(f'./dbpngs/{i}.png', im, format='png')
