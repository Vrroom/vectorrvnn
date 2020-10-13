"""
Plotting functions to debug learned features.
"""
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plotDistanceMatrix (f) : 
    """
    The rows of the matrix f are 
    expected to be the feature vectors. 

    Parameters
    ----------
    f : np.ndarray
        Matrix of features
    """
    dists = distance_matrix(f, f)
    plt.imshow(dists)
    plt.show()

def plotX (x, y, method, **kwargs) : 
    """
    Plot the features along with the 
    classes.

    Parameters
    ----------
    x : np.ndarray
        The features.
    y : np.ndarray
        The classes for each entry in x.
    method : str
        'tsne'/'pca'
    """
    if method == 'tsne' : 
        m = TSNE(n_components=2, perplexity=kwargs['perp'])
    elif method == 'pca' : 
        m = PCA(n_components=2)
    else : 
        raise ValueError
    out = m.fit_transform(x)
    colors = [(1,0,0,0.5), (0, 0, 1, 0.5)]
    for i in range(2) : 
        pts = out[y == i]
        plt.scatter(pts[:, 0], pts[:, 1], color=colors[i])
    plt.show()
