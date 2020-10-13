"""
Fit spline to collection of points. May run into
problems if the points are close by.
"""
import svgpathtools as svg
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from pulp import LpVariable

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

def smoothSpline(points) :
    """
    Create a smooth spline over a set of points. Method
    obtained from:

        https://www.particleincell.com/2012/bezier-splines/

    Examples
    --------
    >>> points = [complex(0, 0), complex(10, 20), complex(20, 20)]
    >>> path = smoothSpline(points)
    >>> points = [path.point(t) for t in np.arange(0, 1, 0.1)]
    >>> x = [p.real for p in points]
    >>> y = [p.imag for p in points]
    >>> plt.plot(x, y)
    >>> plt.show()
    
    Parameters
    ----------
    points: list
        List of points in the complex plane where
        the real coordinate is the x-coordinate 
        and the imaginary coordinate is the y-coordinate
    """
    def linearSystem() : 
        # Prepare linear system of equations.
        eqns = []
        for i in range(n - 1) :
            # First Constraint: P_(i + 1)_1 + P_i_2 = 2K_(i + 1)
            v1x, v2x = f'P_{i + 1}_1_x', f'P_{i}_2_x'
            v1y, v2y = f'P_{i + 1}_1_y', f'P_{i}_2_y'
            ki_1 = points[i+1]
            eqns.append(V[v1x] + V[v2x] == 2 * ki_1.real)
            eqns.append(V[v1y] + V[v2y] == 2 * ki_1.imag)

            # Second Constraint:
            # -2P_(i + 1)_1 + P_(i + 1)_2 - P_i_1 + 2P_i_2 = 0
            v3x, v4x = f'P_{i + 1}_2_x', f'P_{i}_1_x'
            v3y, v4y = f'P_{i + 1}_2_y', f'P_{i}_1_y'
            eqns.append(-2*V[v1x] + V[v3x] - V[v4x] + 2*V[v2x] == 0)
            eqns.append(-2*V[v1y] + V[v3y] - V[v4y] + 2*V[v2y] == 0)

        if abs(points[0] - points[-1]) < 1e-3 : 
            # First Constraint: P_(n - 1)_1 + P_0_2 = 2K_(0)
            v1x, v2x = f'P_0_1_x', f'P_{n-1}_2_x'
            v1y, v2y = f'P_0_1_y', f'P_{n-1}_2_y'
            k0 = points[0]
            eqns.append(V[v1x] + V[v2x] == 2 * k0.real)
            eqns.append(V[v1y] + V[v2y] == 2 * k0.imag)

            # Second Constraint:
            # -2P_0_1 + P_0_2 - P_(n - 1)_1 + 2P_(n - 1)_2 = 0
            v3x, v4x = f'P_0_2_x', f'P_{n-1}_1_x'
            v3y, v4y = f'P_0_2_y', f'P_{n-1}_1_y'
            eqns.append(-2*V[v1x] + V[v3x] - V[v4x] + 2*V[v2x] == 0)
            eqns.append(-2*V[v1y] + V[v3y] - V[v4y] + 2*V[v2y] == 0)
        else : 
            # 4 Boundary Condition constraints for open paths
            # 2P_0_1 - P_0_2 = K_0
            k0 = points[0]
            eqns.append(2*V['P_0_1_x'] - V['P_0_2_x'] == k0.real)
            eqns.append(2*V['P_0_1_y'] - V['P_0_2_y'] == k0.imag)

            # 2P_(n - 1)_2 - P_(n - 1)_1 = K_n
            kn = points[-1]
            eqns.append(2*V[f'P_{n-1}_2_x'] - V[f'P_{n-1}_1_x'] == kn.real)
            eqns.append(2*V[f'P_{n-1}_2_y'] - V[f'P_{n-1}_1_y'] == kn.imag)
        
        return constructLinearSystem(eqns, V.values(), csc_matrix, np.array)

    def makePathFromSolutionVector (x) :
        beziers = []
        for i, knots in enumerate(zip(points, points[1:])) :
            start, end = knots
            P_i_1_x, P_i_1_y = V[f'P_{i}_1_x'], V[f'P_{i}_1_y']
            P_i_2_x, P_i_2_y = V[f'P_{i}_2_x'], V[f'P_{i}_2_y']
            cp1 = complex(x[VIdx[P_i_1_x]], x[VIdx[P_i_1_y]])
            cp2 = complex(x[VIdx[P_i_2_x]], x[VIdx[P_i_2_y]])
            beziers.append(svg.CubicBezier(start, cp1, cp2, end))
        return svg.Path(*beziers)
    
    def initVariables () :
        for i in range(n) : 
            for j in range(1, 3) : 
                xName, yName = f'P_{i}_{j}_x', f'P_{i}_{j}_y'
                V[xName] = LpVariable(xName)
                V[yName] = LpVariable(yName)

    n = len(points) - 1
    V = dict()
    initVariables()
    VIdx = dict(zip(V.values(), range(len(V))))
    A, b = linearSystem()
    x = spsolve(A, b)
    return makePathFromSolutionVector(x)
