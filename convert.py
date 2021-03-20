import pickle
from TripletSVGData import *
import os
import os.path as osp
from osTools import *
from tqdm import tqdm
import multiprocessing as mp

INDIR  = '/misc/extra/data/sumitc/suggeroPicklesIndividual'
OUTDIR = '/misc/extra/data/sumitc/suggeroData'

FILES = list(listdir(INDIR))

def writeOutFile (fname) : 
    with open(fname, 'rb') as fp : 
        tree = pickle.load(fp)
    tree.write(OUTDIR)

with mp.Pool() as p : 
    list(tqdm(p.imap(writeOutFile, FILES, chunksize=100), total=len(FILES)))
