from vectorrvnn.utils import * 
from .data import *

class TripletDataset () : 

    def __init__ (self, datadir) : 
        """
        The datadir will have a pickle file
        for the hierarchy and an svg file. It may 
        have some other metadata but we don't 
        care about that.
        """
        files = listdir(datadir)
        svgFiles, treeFiles = [], []
        self.metadata = []
        for f in files : 
            exampleFiles = listdir(f)
            svgFile  = next(filter(
                lambda x : x.endswith('svg'), 
                exampleFiles
            ))
            treeFile = next(filter(
                lambda x : x.endswith('pkl'), 
                exampleFiles
            ))
            txtFile = next(filter(
                lambda x : x.endswith('txt'),
                exampleFiles
            ))
            with open(txtFile) as fd : 
                self.metadata.append(fd.read().strip())
            svgFiles.append(svgFile)
            treeFiles.append(treeFile)
        self.data = [SVGData(svgFile=sf, treePickle=tf) 
                for sf, tf in zip(svgFiles, treeFiles)]

    def __getitem__ (self, i) :
        return self.data[i]

    def __len__(self) :
        return len(self.data)
