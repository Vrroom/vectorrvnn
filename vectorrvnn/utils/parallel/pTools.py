from tqdm import tqdm
import multiprocessing as mp

def pmap(function, items, chunksize=None) : 
    """ parallel mapper using Pool with progress bar """
    cpu_count = mp.cpu_count()
    if chunksize is None : 
        chunksize = len(items) // (cpu_count * 5)
    chunksize = max(1, chunksize)
    with mp.Pool(cpu_count) as p : 
        mapper = p.imap(function, items, chunksize=chunksize)
        return list(tqdm(mapper, total=len(items)))

def runFnSafely (fn, args, timeout) :
    """
    Runs a function safely.

    Runs the function as a seperate process. The function
    produces some side-effect (possibly by saving stuff
    to the disk). But it may get stuck. This function makes
    sure that all this drama happens in a seperate process
    so that we aren't bothered.
    """
    p = mp.Process(target=fn, args=args)
    p.start()
    p.join(timeout)
