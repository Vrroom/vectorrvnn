# Copyright (c) 2002-2005, Jean-Sebastien Roy (js@jeannot.org)
# Modifications Copyright (c) 2007- Stuart Anthony Mitchell (s.mitchell@auckland.ac.nz)
# $Id: pulp.py 1791 2008-04-23 22:54:34Z smit023 $

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys
import warnings
from collections.abc import Iterable 
from collections import OrderedDict
import re
from scipy.sparse import csc_matrix
class Element(object):
    """Base class for Variable and ConstraintVar
    """
    #to remove illegal characters from the names
    illegal_chars = "-+[] ->/"
    expression = re.compile("[{}]".format(re.escape(illegal_chars)))
    trans = maketrans(illegal_chars, "________")

    def setName(self, name):
        if name:
            if self.expression.match(name):
                warnings.warn("The name {} has illegal characters that will be replaced by _".format(name))
            self.__name = str(name).translate(self.trans)
        else:
            self.__name = None

    def getName(self):
        return self.__name
    name = property(fget = getName,fset = setName)

    def __init__(self, name):
        self.name = name
         # self.hash MUST be different for each variable
        # else dict() will call the comparison operators that are overloaded
        self.hash = id(self)
        self.modified = True

    def __hash__(self):
        return self.hash

    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

    def __neg__(self):
        return - AffineExpression(self)

    def __pos__(self):
        return self

    def __bool__(self):
        return 1

    def __add__(self, other):
        return AffineExpression(self) + other

    def __radd__(self, other):
        return AffineExpression(self) + other

    def __sub__(self, other):
        return AffineExpression(self) - other

    def __rsub__(self, other):
        return other - AffineExpression(self)

    def __mul__(self, other):
        return AffineExpression(self) * other

    def __rmul__(self, other):
        return AffineExpression(self) * other

    def __div__(self, other):
        return AffineExpression(self)/other

    def __rdiv__(self, other):
        raise TypeError("Expressions cannot be divided by a variable")

    def __le__(self, other):
        return AffineExpression(self) <= other

    def __ge__(self, other):
        return AffineExpression(self) >= other

    def __eq__(self, other):
        return AffineExpression(self) == other

    def __ne__(self, other):
        if isinstance(other, Variable):
            return self.name is not other.name
        elif isinstance(other, AffineExpression):
            if other.isAtomic():
                return self is not other.atom()
            else:
                return 1
        else:
            return 1

class Variable(Element):
    """
    This class models an LP Variable with the specified associated parameters

    :param name: The name of the variable used in the output .lp file
    :param lowBound: The lower bound on this variable's range.
        Default is negative infinity
    :param upBound: The upper bound on this variable's range.
        Default is positive infinity
    :param cat: The category this variable is in, Integer, Binary or
        Continuous(default)
    :param e: Used for column based modelling: relates to the variable's
        existence in the objective function and constraints
    """
    def __init__(self, name) : 
        Element.__init__(self,name)
        self.varValue = None
        self.dj = None

    def to_dict(self):
        """
        Exports a variable into a dictionary with its relevant information

        :return: a dictionary with the variable information
        :rtype: dict
        """
        return dict(varValue=self.varValue, dj=self.dj, name=self.name)

    @classmethod
    def from_dict(cls, dj=None, varValue=None, **kwargs):
        """
        Initializes a variable object from information that comes from a dictionary (kwargs)

        :param dj: shadow price of the variable
        :param float varValue: the value to set the variable
        :param kwargs: arguments to initialize the variable
        :return: a :py:class:`Variable`
        :rtype: :Variable
        """
        var = cls(**kwargs)
        var.dj = dj
        var.varValue = varValue
        return var

    def add_expression(self,e):
        self.expression = e
        self.addVariableToConstraints(e)

    def dicts(self, name, indexs, indexStart = []):
        """
        This function creates a dictionary of :py:class:`Variable` with the specified associated parameters.

        :param name: The prefix to the name of each LP variable created
        :param indexs: A list of strings of the keys to the dictionary of LP
            variables, and the main part of the variable name itself
        :param lowBound: The lower bound on these variables' range. Default is
            negative infinity
        :param upBound: The upper bound on these variables' range. Default is
            positive infinity
        :param cat: The category these variables are in, Integer or
            Continuous(default)

        :return: A dictionary of :py:class:`Variable`
        """
        if not isinstance(indexs, tuple): indexs = (indexs,)
        if "%" not in name: name += "_%s" * len(indexs)

        index = indexs[0]
        indexs = indexs[1:]
        d = {}
        if len(indexs) == 0:
            for i in index:
                d[i] = Variable(name % tuple(indexStart + [str(i)]))
        else:
            for i in index:
                d[i] = Variable.dicts(name, indexs, indexStart + [i])
        return d
    dicts = classmethod(dicts)

    def dict(self, name, indexs):
        if not isinstance(indexs, tuple): indexs = (indexs,)
        if "%" not in name: name += "_%s" * len(indexs)

        lists = indexs

        if len(indexs)>1:
            # Cartesian product
            res = []
            while len(lists):
                first = lists[-1]
                nres = []
                if res:
                    if first:
                        for f in first:
                            nres.extend([[f]+r for r in res])
                    else:
                        nres = res
                    res = nres
                else:
                    res = [[f] for f in first]
                lists = lists[:-1]
            index = [tuple(r) for r in res]
        elif len(indexs) == 1:
            index = indexs[0]
        else:
            return {}

        d = {}
        for i in index:
         d[i] = self(name % i)
        return d
    dict = classmethod(dict)

    def value(self):
        return self.varValue

    def valueOrDefault(self):
        return self.varValue if self.varValue != None else 0

    def __ne__(self, other):
        if isinstance(other, Element):
            return self.name is not other.name
        elif isinstance(other, AffineExpression):
            if other.isAtomic():
                return self is not other.atom()
            else:
                return 1
        else:
            return 1

    def addVariableToConstraints(self,e):
        """adds a variable to the constraints indicated by
        the ConstraintVars in e
        """
        for constraint, coeff in e.items():
            constraint.addVariable(self,coeff)

class AffineExpression(OrderedDict):
    """
    A linear combination of :class:`Variables<Variable>`.
    Can be initialised with the following:

    #.   e = None: an empty Expression
    #.   e = dict: gives an expression with the values being the coefficients of the keys (order of terms is undetermined)
    #.   e = list or generator of 2-tuples: equivalent to dict.items()
    #.   e = Element: an expression of length 1 with the coefficient 1
    #.   e = other: the constant is initialised as e

    Examples:

       >>> f=AffineExpression(Element('x'))
       >>> f
       1*x + 0
       >>> x_name = ['x_0', 'x_1', 'x_2']
       >>> x = [Variable(x_name[i], lowBound = 0, upBound = 10) for i in range(3) ]
       >>> c = AffineExpression([ (x[0],1), (x[1],-3), (x[2],4)])
       >>> c
       1*x_0 + -3*x_1 + 4*x_2 + 0
    """
    #to remove illegal characters from the names
    trans = maketrans("-+[] ","_____")
    def setName(self,name):
        if name:
            self.__name = str(name).translate(self.trans)
        else:
            self.__name = None

    def getName(self):
        return self.__name

    name = property(fget=getName, fset=setName)

    def __init__(self, e = None, constant = 0, name = None):
        self.name = name
        # TODO remove isinstance usage
        if e is None:
            e = {}
        if isinstance(e, AffineExpression):
            # Will not copy the name
            self.constant = e.constant
            super(AffineExpression, self).__init__(list(e.items()))
        elif isinstance(e, dict):
            self.constant = constant
            super(AffineExpression, self).__init__(list(e.items()))
        elif isinstance(e, Iterable):
            self.constant = constant
            super(AffineExpression, self).__init__(e)
        elif isinstance(e, Element):
            self.constant = 0
            super(AffineExpression, self).__init__( [(e, 1)])
        else:
            self.constant = e
            super(AffineExpression, self).__init__()

    # Proxy functions for variables

    def isAtomic(self):
        return len(self) == 1 and self.constant == 0 and list(self.values())[0] == 1

    def isNumericalConstant(self):
        return len(self) == 0

    def atom(self):
        return list(self.keys())[0]

    # Functions on expressions

    def __bool__(self):
        return (float(self.constant) != 0.0) or (len(self) > 0)

    def value(self):
        s = self.constant
        for v,x in self.items():
            if v.varValue is None:
                return None
            s += v.varValue * x
        return s

    def valueOrDefault(self):
        s = self.constant
        for v,x in self.items():
            s += v.valueOrDefault() * x
        return s

    def addterm(self, key, value):
            y = self.get(key, 0)
            if y:
                y += value
                self[key] = y
            else:
                self[key] = value

    def emptyCopy(self):
        return AffineExpression()

    def copy(self):
        """Make a copy of self except the name which is reset"""
        # Will not copy the name
        return AffineExpression(self)

    def __str__(self, constant = 1):
        s = ""
        for v in self.sorted_keys():
            val = self[v]
            if val<0:
                if s != "": s += " - "
                else: s += "-"
                val = -val
            elif s != "": s += " + "
            if val == 1: s += str(v)
            else: s += str(val) + "*" + str(v)
        if constant:
            if s == "":
                s = str(self.constant)
            else:
                if self.constant < 0: s += " - " + str(-self.constant)
                elif self.constant > 0: s += " + " + str(self.constant)
        elif s == "":
            s = "0"
        return s

    def sorted_keys(self):
        """
        returns the list of keys sorted by name
        """
        result = [(v.name, v) for v in self.keys()]
        result.sort()
        result = [v for _, v in result]
        return result

    def __repr__(self):
        l = [str(self[v]) + "*" + str(v)
                        for v in self.sorted_keys()]
        l.append(str(self.constant))
        s = " + ".join(l)
        return s

    @staticmethod
    def _count_characters(line):
        #counts the characters in a list of strings
        return sum(len(t) for t in line)

    def addInPlace(self, other):
        if isinstance(other,int) and (other == 0): 
            return self
        if other is None: return self
        if isinstance(other,Element):
            self.addterm(other, 1)
        elif isinstance(other,AffineExpression):
            self.constant += other.constant
            for v,x in other.items():
                self.addterm(v, x)
        elif isinstance(other,dict):
            for e in other.values():
                self.addInPlace(e)
        elif (isinstance(other,list)
              or isinstance(other, Iterable)):
           for e in other:
                self.addInPlace(e)
        else:
            self.constant += other
        return self

    def subInPlace(self, other):
        if isinstance(other,int) and (other == 0): 
            return self
        if other is None: return self
        if isinstance(other,Element):
            self.addterm(other, -1)
        elif isinstance(other,AffineExpression):
            self.constant -= other.constant
            for v,x in other.items():
                self.addterm(v, -x)
        elif isinstance(other,dict):
            for e in other.values():
                self.subInPlace(e)
        elif (isinstance(other,list)
              or isinstance(other, Iterable)):
            for e in other:
                self.subInPlace(e)
        else:
            self.constant -= other
        return self

    def __neg__(self):
        e = self.emptyCopy()
        e.constant = - self.constant
        for v,x in self.items():
            e[v] = - x
        return e

    def __pos__(self):
        return self

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self.copy().addInPlace(other)

    def __iadd__(self, other):
        return self.addInPlace(other)

    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __isub__(self, other):
        return (self).subInPlace(other)

    def __mul__(self, other):
        e = self.emptyCopy()
        if isinstance(other, AffineExpression):
            e.constant = self.constant * other.constant
            if len(other):
                if len(self):
                    raise TypeError("Non-constant expressions cannot be multiplied")
                else:
                    c = self.constant
                    if c != 0:
                        for v,x in other.items():
                            e[v] = c * x
            else:
                c = other.constant
                if c != 0:
                    for v,x in self.items():
                        e[v] = c * x
        elif isinstance(other,Variable):
            return self * AffineExpression(other)
        else:
            if other != 0:
                e.constant = self.constant * other
                for v,x in self.items():
                    e[v] = other * x
        return e

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other,AffineExpression) or isinstance(other,Variable):
            if len(other):
                raise TypeError("Expressions cannot be divided by a non-constant expression")
            other = other.constant
        e = self.emptyCopy()
        e.constant = self.constant / other
        for v,x in self.items():
            e[v] = x / other
        return e

    def __truediv__(self, other):
        if isinstance(other,AffineExpression) or isinstance(other,Variable):
            if len(other):
                raise TypeError("Expressions cannot be divided by a non-constant expression")
            other = other.constant
        e = self.emptyCopy()
        e.constant = self.constant / other
        for v,x in self.items():
            e[v] = x / other
        return e

    def __rdiv__(self, other):
        e = self.emptyCopy()
        if len(self):
            raise TypeError("Expressions cannot be divided by a non-constant expression")
        c = self.constant
        if isinstance(other,AffineExpression):
            e.constant = other.constant / c
            for v,x in other.items():
                e[v] = x / c
        else:
            e.constant = other / c
        return e

    def __le__(self, other):
        return Constraint(self - other, const.ConstraintLE)

    def __ge__(self, other):
        return Constraint(self - other, const.ConstraintGE)

    def __eq__(self, other):
        return Constraint(self - other, const.ConstraintEQ)

    def to_dict(self):
        """
        exports the :py:class:`AffineExpression` into a list of dictionaries with the coefficients
        it does not export the constant

        :return: list of dictionaries with the coefficients
        :rtype: list
        """
        return [dict(name=k.name, value=v) for k, v in self.items()]


class Constraint(AffineExpression):
    """An LP constraint"""
    def __init__(self, e = None, name = None, rhs = None):
        """
        :param e: an instance of :class:`AffineExpression`
        :param name: identifying string
        :param rhs: numerical value of constraint target
        """
        AffineExpression.__init__(self, e, name = name)
        if rhs is not None:
            self.constant -= rhs
        self.pi = None
        self.slack = None
        self.modified = True

    def __str__(self):
        s = AffineExpression.__str__(self, 0)
        if self.sense is not None:
            s += " = " + str(-self.constant)
        return s

    def changeRHS(self, RHS):
        """
        alters the RHS of a constraint so that it can be modified in a resolve
        """
        self.constant = -RHS
        self.modified = True

    def __repr__(self):
        s = AffineExpression.__repr__(self)
        s += " = 0"
        return s

    def copy(self):
        """Make a copy of self"""
        return Constraint(self)

    def emptyCopy(self):
        return Constraint()

    def addInPlace(self, other):
        if isinstance(other,Constraint):
            AffineExpression.addInPlace(self, other)
        elif isinstance(other,list):
            for e in other:
                self.addInPlace(e)
        else:
            AffineExpression.addInPlace(self, other)
        return self

    def subInPlace(self, other):
        if isinstance(other,Constraint):
            AffineExpression.subInPlace(self, other)
        elif isinstance(other,list):
            for e in other:
                self.subInPlace(e)
        else:
            AffineExpression.subInPlace(self, other)
        return self

    def __neg__(self):
        c = AffineExpression.__neg__(self)
        return c

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self.copy().addInPlace(other)

    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __mul__(self, other):
        if isinstance(other, Constraint):
            c = AffineExpression.__mul__(self, other)
            return c
        else:
            return AffineExpression.__mul__(self, other)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other,Constraint):
            c = AffineExpression.__div__(self, other)
            return c
        else:
            return AffineExpression.__div__(self, other)

    def __rdiv__(self, other):
        if isinstance(other,Constraint):
            c = AffineExpression.__rdiv__(self, other)
            return c
        else:
            return AffineExpression.__rdiv__(self, other)

    def to_dict(self):
        """
        exports constraint information into a dictionary

        :return: dictionary with all the constraint information
        """
        return dict(constant=self.constant,
                    name=self.name,
                    coefficients=AffineExpression.to_dict(self))

    @classmethod
    def from_dict(cls, _dict):
        """
        Initializes a constraint object from a dictionary with necessary information

        :param dict _dict: dictionary with data
        :return: a new :py:class:`Constraint`
        """
        const = cls(e=_dict['coefficients'], rhs=-_dict['constant'], name=_dict['name'])
        return const

def constructLinearSystem (constraints, variables,
    matrixConstructor, vectorConstructor) :

    def insert (d, r, c) : 
        data.append(d)
        rows.append(r)
        cols.append(c)

    data, rows, cols = [], [], []
    varIdx = dict(zip(variables, range(len(variables))))
    b = []
    for i, constraint in enumerate(constraints) :
        for v in constraint.keys() :
            insert(constraint[v], i, varIdx[v])
        b.append(-constraint.constant)
    A = matrixConstructor((data, (rows, cols)))
    b = vectorConstructor(b)
    return A, b
