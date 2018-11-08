import numpy as np

class Spec:
    def __init__(self, spaces, name=None):
        self.name, self.spaces = name, spaces

    def __repr__(self):
        return "Spec: %s\n%s" % (self.name, "\n".join(map(str, self.spaces)))


class Space:
    def __init__(self, shape=(), dtype=np.int32, domain=(0, 1), categorical=False, name=None):
        self.name = name
        self.shape, self.dtype = shape, dtype
        self.categorical, (self.lo, self.hi) = categorical, domain

    def sample(self, n=1):
        if np.issubdtype(self.dtype, np.integer):
            return np.random.randint(self.lo, self.hi+1, (n, ) + self.shape)

        if np.issubdtype(self.dtype, np.float):
            return np.random.uniform(self.lo, self.hi+1e-10, (n, ) + self.shape)

    def __repr__(self):
        return "Space(%s, %s, %s)" % (self.name, str(self.shape), str(self.dtype).strip("<class>' "))
