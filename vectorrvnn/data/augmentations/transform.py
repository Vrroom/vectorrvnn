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

class GraphicCompose (SVGDataTransform) : 

    def __init__ (self, p=1.0) : 
        super(GraphicCompose, self).__init__(p=p)

    def sample (self, r1, r2) : 
        """ 
        Sample two non overlapping boxes to fit both 
        graphics in a 100 by 100 box.

        rs are the ratio ws / hs for s = 1, 2
        """
        bigBox = BBox(0, 0, 100, 100, 100, 100)
        for i in range(10) : 
            x1, y1, x2, y2 = [trng.randint(0, 95) for _ in range(4)]
            h1, h2 = [trng.randint(5, 50) for _ in range(2)]
            w1, w2 = r1 * h1, r2 * h2
            box1 = BBox(x1, y1, x1 + w1, y1 + h1, w1, h1)
            box2 = BBox(x2, y2, x2 + w2, y2 + h2, w2, h2)
            if box1 in bigBox and box2 in bigBox and (box1 ^ box2) : 
                return box1, box2
        return None
    
    def _calculateR (self, pt) :
        box = pt.nodes[findRoot(pt)]['bbox']
        return box.w / box.h

    def _fitInBox (self, pt, box) : 
        graphicBox = pt.nodes[findRoot(pt)]['bbox']
        center = graphicBox.center()
        pt = translate(pt, -center.real, -center.imag)
        s = box.w / graphicBox.w
        pt = scale(pt, s)
        pt = translate(pt, box.center().real, box.center().imag)
        return pt

    def transform (self, svgdata, *args) : 
        other = trng.choice(list(args[0]))
        r1, r2 = self._calculateR(svgdata), self._calculateR(other)
        boxes = self.sample(r1, r2)
        if boxes is None : 
            return svgdata
        box1, box2 = boxes
        svgdata = self._fitInBox(svgdata, box1)
        other = self._fitInBox(other, box2)
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
