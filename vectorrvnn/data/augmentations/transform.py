from vectorrvnn.utils import *

class SVGDataTransform : 

    def __init__ (self, transform=None, p=1.0) : 
        if transform is None : 
            self.transform = lambda *args : args[0]
        else : 
            self.transform = transform
        self.p = p
    
    def __call__ (self, svgdata, *args) : 
        toss = random.random()
        if toss < self.p :
            return self.transform(svgdata, *args)
        else :
            return svgdata

class NoFill (SVGDataTransform) : 
    
    def transform (self, svgdata, *args) : 
        svgdata_ = deepcopy(svgdata)
        svgdata_.doc = modAttr(svgdata_.doc, 'fill', 'none')
        return svgdata_

class Rotate (SVGDataTransform) : 
    """ Randomly rotate graphic about it's center """     
    def __init__ (self, degreeRange=(-180, 180), p=1.0) : 
        super(Rotate, self).__init__(p=p)
        self.degreeRange = degreeRange

    def transform (self, svgdata, *args) : 
        svgdata_ = deepcopy(svgdata)
        center = reduce(
            lambda x, y : x + y, 
            map(
                pathBBox, 
                cachedPaths(svgdata.doc)
            )
        ).center()
        degree = random.randint(*degreeRange)
        svgdata_.doc = rotate(
            svgdata_.doc, 
            degree, 
            px=center.real, 
            py=center.imag
        )
        return svgdata_


