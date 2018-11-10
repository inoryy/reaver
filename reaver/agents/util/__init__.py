import numpy as np
from .logger import AgentLogger


def discounted_cumsum(x, discount):
    y = np.zeros_like(x)
    y[-1] = x[-1]
    for t in range(x.shape[0]-2, -1, -1):
        y[t] = x[t] + discount[t] * y[t+1]
    return y
