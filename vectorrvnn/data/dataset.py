from vectorrvnn.utils import * 
from .data import *
import re
import warnings

class SVGDataset () : 

    def __init__ (self, datadir, idsToAvoid=[]) : 
        """
        The datadir will have a pickle file
        for the hierarchy and an svg file. It may 
        have some other metadata but we don't care about that.
        """
        files = listdir(datadir)
        svgFiles, treeFiles = [], []
        self.ids = []
        ignored = 0
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
                id = int(re.findall(r'\d+', fd.read())[0])

            if id not in idsToAvoid :
                self.ids.append(id)
                svgFiles.append(svgFile)
                treeFiles.append(treeFile)
            else : 
                ignored += 1

        if ignored > 0 :
            warnings.warn(f'Ignored {ignored} graphics')

        self.data = []
        for sf, tf in zip(svgFiles, treeFiles) :
            try : 
                self.data.append(SVGData(svgFile=sf, treePickle=tf)) 
            except Exception :
                pass

    def __getitem__ (self, i) :
        return self.data[i]

    def __len__(self) :
        return len(self.data)
