from tqdm import tqdm
import multiprocessing as mp
from functools import wraps
import os
import hashlib

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

def skip_if_processed(task_file="completed_tasks.txt"):
    """
    Decorator to skip function execution if it has been previously processed. 

    This is useful in data processing if program crashes and you want to restart
    without doing everything all over again.
    """

    # Initialize a set to hold completed task hashes
    completed_task_hashes = set()

    # Load completed task hashes from file
    if os.path.exists(task_file):
        with open(task_file, "r") as f:
            completed_task_hashes = set(line.strip() for line in f.readlines())

    def compute_hash(args):
        """Compute SHA-256 hash for the given arguments."""
        args_str = str(args)
        return hashlib.sha256(args_str.encode()).hexdigest()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_hash = compute_hash(args)

            # If the task is not yet completed, run it and mark as completed
            if task_hash not in completed_task_hashes:
                result = func(*args, **kwargs)

                # Update completed task hashes and save to file
                completed_task_hashes.add(task_hash)
                with open(task_file, "a") as f:
                    f.write(f"{task_hash}\n")

                return result
            else:
                print(f"Task with hash {task_hash} has already been processed, skipping.")
        return wrapper
    return decorator
