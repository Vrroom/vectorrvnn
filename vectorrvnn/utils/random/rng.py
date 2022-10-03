import random
import string

rng = random.Random(1000)

def random_string(k=10):
    return ''.join(rng.choices(string.ascii_lowercase, k=k))
