from tqdm import tqdm
import multiprocessing as mp
from functools import wraps

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

def safe_run(f):
    """
    Sometimes in life, you aren't bothered whether your function fails, 
    you just want to run it on a lot of inputs. This decorator let's you 
    annotate your function as such and have it run without crashing your
    program

    Usage:

    @safe_run
    def potentially_dubious_fn(x):
        ...
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f"Exception occurred while running {f.__name__}: {e}")
            return None
    return wrapped
