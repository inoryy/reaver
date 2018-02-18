import numpy as np
from common.config import Config


def flatten(x):
    x = np.array(x) # TODO replace with concat if axis != 0
    return x.reshape(-1, *x.shape[2:])


def flatten_dicts(x):
    return {k: flatten([s[k] for s in x]) for k in x[0].keys()}


#  n-steps x actions x envs -> actions x n-steps*envs
def flatten_lists(x):
    return [flatten([s[a] for s in x]) for a in range(len(x[0]))]