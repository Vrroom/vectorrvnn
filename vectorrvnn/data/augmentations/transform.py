from vectorrvnn.utils import *

class SVGDataTransform : 

    def __init__ (self, p=1.0) : 
        self.p = p

    def transform (self, svgdata, *args) : 
        return svgdata
    
    def __call__ (self, svgdata, *args) : 
        toss = random.random()
        if toss < self.p :
            return self.transform(svgdata, *args)
        else :
            return svgdata

class NoFill (SVGDataTransform) : 
    
    def transform (self, svgdata, *args) : 
        svgdata_ = deepcopy(svgdata)
        svgdata_.doc = modAttrs(
            svgdata_.doc, 
            dict(
                fill='none', 
                stroke='black'
            )
        )
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
                [p.path for p in cachedPaths(svgdata.doc)]
            )
        ).center()
        degree = random.randint(*self.degreeRange)
        svgdata_.doc = rotate(
            svgdata_.doc, 
            degree, 
            px=center.real, 
            py=center.imag
        )
        return svgdata_


