import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
from functools import reduce

def pathBBox (path) : 
    try : 
        x, X, y, Y = path.bbox()
    except Exception : 
        x, X, y, Y = 0, 0, 0, 0
    return BBox(x, y, X, Y, X - x, Y - y)

def getDocBBox (doc) :
    x, y, w, h = doc.get_viewbox()
    return BBox(x, y, x + w, y + h, w, h)

def setDocBBox (doc, box) : 
    string = ' '.join(map(str, box.tolist()))
    doc.set_viewbox(string)

def pathBBoxTooSmall (pathbox, docbox) : 
    rel = pathbox / docbox
    return rel.normalized().area() <= 5e-4

def union(boxes) : 
    return reduce(lambda x, y: x | y, boxes)

def intersection (boxes) : 
    return reduce(lambda x, y: x & y, boxes)

def pathsetBox (t, ps) : 
    return union([t.nodes[i]['bbox'] for i in ps])

class BBox : 

    def __init__ (self, x, y, X, Y, w, h) : 
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        self.w = w
        self.h = h
        self.assertConsistent()

    def assertConsistent (self) : 
        assert(self.X >= self.x)
        assert(self.Y >= self.y)
        assert(np.isclose(self.X - self.x, self.w))
        assert(np.isclose(self.Y - self.y, self.h))

    def iou (self, that) : 
        if (self + that).isDegenerate() \
                or (self * that).isDegenerate() :
            return 0
        intersection = (self * that).area()
        union = (self + that).area()
        return intersection / union

    def isDegenerate (self) : 
        return np.isclose(self.w, 0) and np.isclose(self.h, 0)

    def area (self) : 
        if self.isDegenerate() : 
            return 0
        else : 
            return self.w * self.h

    def center (self) : 
        return complex(
            (self.x + self.X) / 2, 
            (self.y + self.Y) / 2
        )

    def __eq__ (self, that) : 
        return self.x == that.x \
                and self.X == that.X \
                and self.y == that.y \
                and self.Y == that.Y

    def __mul__ (self, s) : 
        return self.scaled(s, origin='center')

    def __or__ (self, that) : 
        x = min(self.x, that.x)
        y = min(self.y, that.y)
        X = max(self.X, that.X)
        Y = max(self.Y, that.Y)
        return BBox(x, y, X, Y, X - x, Y - y)

    def __and__ (self, that) : 
        x = max(self.x, that.x)
        y = max(self.y, that.y)
        X = min(self.X, that.X)
        Y = min(self.Y, that.Y)
        return BBox(x, y, X, Y, X - x, Y - y)

    def __contains__ (self, that) :
        return (self.x <= that.x <= that.X <= self.X \
                and self.y <= that.y <= that.Y <= self.Y) \
                and not self == that

    def __truediv__ (self, that): 
        nx = (self.x - that.x) / that.w
        ny = (self.y - that.y) / that.h
        nX = (self.X - that.x) / that.w
        nY = (self.Y - that.y) / that.h
        nw = nX - nx
        nh = nY - ny
        return BBox(nx, ny, nX, nY, nw, nh)

    def normalized (self) : 
        d = max(self.w, self.h)
        nx = self.x - (d - self.w) / 2
        ny = self.y - (d - self.h) / 2
        nX = nx + d
        nY = ny + d
        return BBox(nx, ny, nX, nY, d, d)

    def tolist (self) : 
        return [self.x, self.y, self.w, self.h]

    def __repr__ (self) : 
        x = self.x
        y = self.y
        X = self.X
        Y = self.Y
        w = self.w
        h = self.h
        return f'BBox(x={x}, y={y}, X={X}, Y={Y}, w={w}, h={h})'

    def __xor__ (self, that) : 
        """ check whether boxes are disjoint """
        b1 = sg.box(self.x, self.y, self.X, self.Y)
        b2 = sg.box(that.x, that.y, that.X, that.Y)
        return b1.disjoint(b2)

    def rotated (self, degree, pt=None) : 
        if pt is None:  
            pt = sg.Point(0, 0)
        else : 
            pt = sg.Point(pt.real, pt.imag)
        x, y, X, Y = sa.rotate(self.toShapely(), degree, origin=pt).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def translated (self, tx, ty=0) : 
        x, y, X, Y = sa.translate(self.toShapely(), tx, ty).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def scaled (self, sx, sy=None, origin=sg.Point(0, 0)) : 
        if sy is None : 
            sy = sx
        x, y, X, Y = sa.scale(self.toShapely(), sx, sy, origin=origin).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def skewX (self, xs) : 
        x, y, X, Y = sa.skew(self.toShapely(), xs=xs).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def skewY (self, ys) : 
        x, y, X, Y = sa.skew(self.toShapely(), ys=ys).bounds
        return BBox(x, y, X, Y, X - x, Y - y)

    def toShapely (self) : 
        return sg.Polygon([
            (self.x, self.y),
            (self.x, self.Y),
            (self.X, self.Y),
            (self.X, self.y)
        ])

