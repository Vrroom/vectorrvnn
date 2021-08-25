from vectorrvnn.utils import *
from skimage import color
from copy import deepcopy

class SVGDataTransform : 

    def __init__ (self, p=1.0) : 
        self.p = p

    def transform (self, svgdata, *args) : 
        return svgdata
    
    def __call__ (self, svgdata, *args) : 
        toss = rng.uniform(0, 1)
        if toss < self.p :
            return self.transform(svgdata, *args)
        else :
            return svgdata

class NoFill (SVGDataTransform) : 
    
    def transform (self, svgdata, *args) : 
        modAttrs(
            svgdata,
            dict(
                fill=lambda k, x: 'none', 
                stroke=lambda k, x: 'black'
            )
        )
        return svgdata

class HSVJitter (SVGDataTransform) : 

    def __init__ (self, perturbAmt=0.2, p=1.0) : 
        super(HSVJitter, self).__init__(p=p)
        self.perturbAmt = perturbAmt

    def jitterer (self, attr, elt) : 
        try : 
            val = xmlAttributeGet(elt, attr, 'none')
            if val == 'none' : 
                return 'none'
            nprng = np.random.RandomState(rng.randint(0, 10000))
            rgb = np.array(parseColor(val), dtype=np.float)
            hsv = color.rgb2hsv(rgb)
            delta = nprng.uniform(-self.perturbAmt, self.perturbAmt, 3)
            hsv_ = np.clip(hsv + delta, 0, 1)
            r, g, b = (color.hsv2rgb(hsv_) * 255).astype(np.uint8)
            return f'rgb({r}, {g}, {b})'
        except Exception : 
            return 'none'

    def transform(self, svgdata, *args) : 
        modAttrs(
            svgdata, 
            dict(
                stroke=self.jitterer, 
                fill=self.jitterer
            )
        )
        return svgdata

class StrokeWidthJitter (SVGDataTransform) :

    def __init__ (self, scaleRange=(0.7, 1.5), p=1.0) :
        super(StrokeWidthJitter, self).__init__(p=p)
        self.scaleRange=scaleRange

    def jitterer (self, k, e) : 
        val = xmlAttributeGet(e, 'stroke-width', None)
        if val is not None: 
            newVal = float(val) * rng.uniform(*self.scaleRange)
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
        jitterer = lambda k, e : f'{rng.uniform(self.lowerBound, 1):.3f}'
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
        s = rng.uniform(0.5, 1.5)
        theta = rng.uniform(0, 2 * np.pi)
        box2 = box2.scaled(s)
        r1 = abs(box1.center() - complex(box1.x, box1.y))
        r2 = abs(box2.center() - complex(box2.x, box2.y))
        mr = rng.uniform(0, (r1 + r2) / 2)
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
        other = deepcopy(rng.choice(list(args[0])))
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
        degree = rng.randint(*self.degreeRange)
        rotate(svgdata, degree, center)
        return svgdata
