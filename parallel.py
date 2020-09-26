import multiprocessing as mp
import more_itertools
import os
import os.path as osp

def parallelize(indirList, outdir, function, writer, **kwargs) :
    """ 
    Use multiprocessing library to 
    parallelize preprocessing. 

    A very common operation that I
    have to do is to apply some
    operations on some files in a
    directory and write to some other
    directory. Often, these operations 
    are very slow. This function takes
    advantage of the multiple cpus to 
    distribute the computation.

    Parameters
    ----------
    indirList : list
        paths to the input directories
    outdir : str
        path to the output directory
    function : lamdba 
        function to be applied on 
        each file in the input directory
    writer : lambda
        function which writes the results
        of function application to a file of
        given name
    """
    def subTask (inChunk, outChunk) :
        for i, o in zip(inChunk, outChunk) :
            obj = function(i, **kwargs)
            writer(obj, o, **kwargs)
    cpus = mp.cpu_count()
    names = list(map(lambda x : osp.splitext(x)[0], os.listdir(indirList[0])))
    names.sort()
    inFullPaths = list(zipDirs(indirList))
    outFullPaths = [osp.join(outdir, name) for name in names]
    inChunks = more_itertools.divide(cpus, inFullPaths)
    outChunks = more_itertools.divide(cpus, outFullPaths)
    processList = []
    for inChunk, outChunk in zip(inChunks, outChunks) :
        p = mp.Process(target=subTask, args=(inChunk, outChunk,))
        p.start()
        processList.append(p)
    for p in processList :
        p.join()

