from vectorrvnn.utils import *
from .rng import *

class SVGDataTransform : 

    def __init__ (self, p=1.0) : 
        self.p = p

    def transform (self, svgdata, *args) : 
        return svgdata
    
    def __call__ (self, svgdata, *args) : 
        toss = trng.uniform(0, 1)
        if toss < self.p :
            return self.transform(svgdata, *args)
        else :
            return svgdata

class NoFill (SVGDataTransform) : 
    
    def transform (self, svgdata, *args) : 
        modAttrs(
            svgdata,
            dict(
                fill=lambda x: 'none', 
                stroke=lambda x: 'black'
            )
        )
        return svgdata

class StrokeWidthJitter (SVGDataTransform) :

    def __init__ (self, scaleRange=(0.7, 1.5), p=1.0) :
        super(StrokeWidthJitter, self).__init__(p=p)
        self.scaleRange=scaleRange

    def jitterer (self, e) : 
        val = xmlAttributeGet(e, 'stroke-width', None)
        if val is not None: 
            newVal = float(val) * trng.uniform(*self.scaleRange)
            return f'{newVal:.3f}'
        else :
            return None
    
    def transform(self, svgdata, *args):  
        modAttr(
            svgdata, 
            'stroke-width',
            self.jitterer
        )
        return svgdata

class OpacityJitter (SVGDataTransform) : 

    def __init__ (self, lowerBound=0.7, p=1.0) :
        super(OpacityJitter, self).__init__(p=p)
        self.lowerBound=lowerBound
    
    def transform(self, svgdata, *args):  
        jitterer = lambda e : f'{trng.uniform(self.lowerBound, 1):.3f}'
        modAttrs(
            svgdata, 
            {
                "fill-opacity": jitterer, 
                "stroke-opacity": jitterer
            }
        )
        return svgdata


class Rotate (SVGDataTransform) : 
    """ Randomly rotate graphic about it's center """     
    def __init__ (self, degreeRange=(-180, 180), p=1.0) : 
        super(Rotate, self).__init__(p=p)
        self.degreeRange = degreeRange

    def transform (self, svgdata, *args) : 
        rootNode = svgdata.nodes[findRoot(svgdata)]
        center = rootNode['bbox'].center()
        degree = random.randint(*self.degreeRange)
        rotate(svgdata, degree, center)
        return svgdata
