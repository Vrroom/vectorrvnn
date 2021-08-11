from vectorrvnn.utils import *
from copy import deepcopy
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

class GraphicCompose (SVGDataTransform) : 

    def __init__ (self, p=1.0) : 
        super(GraphicCompose, self).__init__(p=p)

    def sample (self, box1, box2) : 
        """ 
        Sample two non overlapping boxes. Since these graphics 
        are going to be normalized anyway, fix the first box 
        and sample the scale and angle of the second box wrt 
        the first. Then sample a distance so that they don't 
        overlap.
        """
        s = trng.uniform(0.5, 1.5)
        theta = trng.uniform(0, 2 * np.pi)
        box2 = box2.scaled(s)
        r1 = abs(box1.center() - complex(box1.x, box1.y))
        r2 = abs(box2.center() - complex(box2.x, box2.y))
        mr = trng.uniform(0, (r1 + r2) / 2)
        center2 = box1.center() + (r1 + r2 + mr) * np.exp(1j * theta)
        tr = center2 - box2.center()
        box2 = box2.translated(tr.real, tr.imag)
        assert box1 ^ box2
        return box1, box2
    
    def _bbox (self, pt) :
        box = pt.nodes[findRoot(pt)]['bbox']
        return box

    def _fitInBox (self, pt, box) : 
        graphicBox = pt.nodes[findRoot(pt)]['bbox']
        center = graphicBox.center()
        pt = translate(pt, -center.real, -center.imag)
        s = box.w / graphicBox.w
        pt = scale(pt, s)
        pt = translate(pt, box.center().real, box.center().imag)
        return pt

    def transform (self, svgdata, *args) : 
        other = deepcopy(trng.choice(list(args[0])))
        box1, box2 = self._bbox(svgdata), self._bbox(other)
        box1, box2 = self.sample(box1, box2)
        svgdata = self._fitInBox(svgdata, box1)
        other   = self._fitInBox(other, box2)
        return svgdata | other

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
