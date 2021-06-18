from functools import reduce, wraps
from collections import namedtuple

def pathBBox (path) : 
    return ExtentBBox(*path.bbox())

def getDocBBox (doc) :
    return DimBBox(*doc.get_viewbox())

def setDocBBox (doc, box) : 
    if isinstance(box, ExtentBBox) :
        box = box.todim()
    lst = [box.x, box.y, box.w, box.h]
    string = ' '.join(map(str, lst))
    doc.set_viewbox(string)

class BBox : 

    def todim (self) : return self
    def toext (self) : return self

    def iou (self, that) : 
        if (self + that).isDegenerate() \
                or (self * that).isDegenerate() :
            return 0
        intersection = (self * that).area()
        union = (self + that).area()
        return intersection / union

class ExtentBBox (BBox):

    def __init__ (self, x, X, y, Y) : 
        self.x = x
        self.X = X
        self.y = y
        self.Y = Y

    def tolist(self) : 
        return [
            self.x,
            self.X,
            self.y,
            self.Y
        ]

    def area (self) : 
        if self.isDegenerate() : 
            return 0
        else : 
            return (self.X - self.x) * (self.Y - self.y)

    def center (self) : 
        return complex(
            (self.x + self.X) / 2, 
            (self.y + self.Y) / 2
        )

    def normalized (self) : 
        w, h = self.X - self.x, self.Y - self.y
        d = max(w, h)
        return ExtentBBox(
            self.x - (d - w) / 2,
            self.x - (d - w) / 2 + d,
            self.y - (d - h) / 2,
            self.y - (d - h) / 2 + d
        )

    def isDegenerate (self) : 
        w, h = self.X - self.x, self.Y - self.y
        return (w < 1e-5 and h < 1e-5) or w < 0 or h < 0

    def __eq__ (self, that) : 
        return self.x == that.x \
                and self.X == that.X \
                and self.y == that.y \
                and self.Y == that.Y

    def __add__(self, that) : 
        return ExtentBBox(
            min(self.x, that.x), 
            max(self.X, that.X), 
            min(self.y, that.y), 
            max(self.Y, that.Y)
        )

    def __mul__ (self, that) : 
        x = max(self.x, that.x)
        y = max(self.y, that.y)
        X = min(self.X, that.X)
        Y = min(self.Y, that.Y)
        return ExtentBBox(x, X, y, Y)
            

    def contains (self, that) :
        return (self.x <= that.x <= that.X <= self.X \
                and self.y <= that.y <= that.Y <= self.Y) \
                and not self == that

    def todim (self) :
        return DimBBox(
            self.x, 
            self.y,
            self.X - self.x,
            self.Y - self.y
        )

class DimBBox (BBox):
    
    def __init__ (self, x, y, w, h) : 
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def tolist(self) : 
        return [
            self.x,
            self.y,
            self.w,
            self.h
        ]

    def area (self) : 
        if self.isDegenerate() :
            return 0
        else : 
            return self.w * self.h

    def center (self) : 
        return complex(
            (self.x + (self.w / 2)), 
            (self.y + (self.h / 2))
        )

    def normalized (self) : 
        d = max(self.w, self.h)
        return DimBBox(
            self.x - (d - self.w) / 2,
            self.y - (d - self.h) / 2,
            d, 
            d
        )

    def isDegenerate (self) : 
        return (self.w < 1e-5 and self.h < 1e-5) \
                or self.w < 0 or self.h < 0

    def __eq__ (self, that) : 
        return self.x == that.x \
                and self.y == that.y \
                and self.w == that.w \
                and self.h == that.h

    def __add__ (self, that) : 
        x = min(self.x, that.x)
        y = min(self.y, that.y)
        X = max(self.x + self.w, that.x + that.w)
        Y = max(self.y + self.h, that.y + that.h)
        return DimBBox(x, y, X - x, Y - y)

    def __mul__ (self, that) : 
        x = max(self.x, that.x)
        y = max(self.y, that.y)
        X = min(self.x + self.w, that.x + that.w)
        Y = min(self.y + self.h, that.y + that.h)
        return DimBBox(x, y, X - x, Y - y)
    
    def contains (self, that) :
        return (self.x <= that.x <= that.x + that.w <= self.x + self.w \
                and self.y <= that.y <= that.y + that.h <= self.y + self.h) \
                and not self == that

    def toext (self) :
        return ExtentBBox(
            self.x,
            self.x + self.w,
            self.y,
            self.y + self.h
        )

