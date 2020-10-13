from spline import smoothSpline
import itertools
import numpy as np
from complexOps import complexDot

def noisyPath (path) : 
    """
    Add Perlin Noise to path.

    Right now, I've done something which works well. 
    I perturb the path at 10 evenly spaced points. Then,
    I interpolate a smooth spline over them.

    Parameters
    ----------
    path: svg.Path
        Path to which noise has to be added.
    """
    xNoise, yNoise = PerlinNoise(seed=0), PerlinNoise(seed=1)
    points = [path.point(t) for t in np.arange(0, 1.01, 0.1)]
    points = [p + complex(xNoise(p), yNoise(p)) for p in points]
    return smoothSpline(points)

class PerlinNoise () :
    """ 
    Implementation of the Perlin Method mentioned at:

        http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf

    to generate well behaved noise in 2D plane

    Examples
    --------
    >>> noise = PerlinNoise(seed=10)
    >>> image = np.zeros((100, 100))
    >>> for i in range(100) :
    >>>     for j in range(100) :
    >>>         image[i, j] = noise(complex(i / 10, j / 10))
    >>> plt.imshow(image)
    >>> plt.show()
    """

    nGrads = 16

    def __init__ (self, seed=0) :
        """
        Constructor.

        Fix the gradients for the lattice points.
        """
        rng = np.random.RandomState(seed)
        self.grads = [self._angleToGrad(rng.uniform(high=2*np.pi)) for _ in range(self.nGrads)]

    def _gradIdx (self, a, b) :
        n = (a * b) % self.nGrads
        return n
    
    def _angleToGrad (self, angle) :
        return complex(np.cos(angle), np.sin(angle))

    def _lattice (self) :
        points = itertools.product(range(2), range(2))
        return points

    def _f (self, t) : 
        return (6 * (t ** 5)) - (15 * (t ** 4)) + (10 * (t ** 3))

    def __call__ (self, p) :
        """
        Given a point somewhere in the 2D plane, 
        find out how much noise is to be added to that point.
        """
        x, y = p.real, p.imag
        if (x.is_integer() and y.is_integer()) :
            return 0
        else : 
            px = int(np.floor(x))
            py = int(np.floor(y))

            relCoord = complex(x - px, y - py)
            u, v = relCoord.real, relCoord.imag

            grads = [self.grads[self._gradIdx(px + p[0], py + p[1])] for p in self._lattice()]
            noises = [complexDot(g, relCoord - complex(*p)) for g, p in zip(grads, self._lattice())]

            nx0 = noises[0] * (1 - self._f(u)) + noises[2] * self._f(u)
            nx1 = noises[1] * (1 - self._f(u)) + noises[3] * self._f(u)

            nxy = nx0 * (1 - self._f(v)) + nx1 * self._f(v)
            return nxy
