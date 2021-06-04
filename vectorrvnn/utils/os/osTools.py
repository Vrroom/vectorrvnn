import os
import os.path as osp

def getBaseName(fullName) : 
    return osp.splitext(osp.split(fullName)[1])[0]

def zipDirs (dirList) :
    """ 
    A common operation is to get SVGs
    or graphs from different directories
    and match them up and perform 
    some operations on them. 

    Parameters
    ----------
    dirList : list
        List of directories to zip.
    """
    filesInDirList = list(map(listdir, dirList))
    return zip(*filesInDirList)

def listdir (path) :
    """
    Convenience function to get 
    full path details while calling os.listdir

    Also ensures that the order is always the same.

    Parameters
    ----------
    path : str
        Path to be listed.
    """
    paths = [osp.join(path, f) for f in os.listdir(path)]
    paths.sort()
    return paths

