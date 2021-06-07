""" https://diglib.eg.org/bitstream/handle/10.2312/egs20211016/029-032.pdf """
from vectorrvnn.utils import *
from vectorrvnn.geometry import *

def containmentGraph (doc) : 
    return relationshipGraph(doc, containsBBox, False)
    
