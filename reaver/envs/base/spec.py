import numpy as np


class Spec:
    def __init__(self, spaces, name=None):
        self.name, self.spaces = name, spaces
        for i, space in enumerate(self.spaces):
            if not space.name:
                space.name = str(i)

    def sample(self, n=1):
        return [space.sample(n) for space in self.spaces]

    def __repr__(self):
        return "Spec: %s\n%s" % (self.name, "\n".join(map(str, self.spaces)))

    def __iter__(self):
        return (space for space in self.spaces)

    def __len__(self):
        return len(self.spaces)


class Space:
    def __init__(self, shape=(), dtype=np.int32, domain=(0, 1), categorical=False, name=None):
        self.name = name
        self.shape, self.dtype = shape, dtype
        self.categorical, (self.lo, self.hi) = categorical, domain

    def is_discrete(self):
        return np.issubdtype(self.dtype, np.integer)

    def is_continuous(self):
        return np.issubdtype(self.dtype, np.floating)

    def is_spatial(self):
        return len(self.shape) > 1 or type(self.hi) in [list, tuple]

    def size(self):
        if self.is_discrete() and self.categorical:
            if self.is_spatial():
                return self.hi
            return self.hi - self.lo

        sz = 1
        if len(self.shape) == 1:
            sz = self.shape[0]

        if self.is_continuous():
            # mu_1, ..., mu_n, log_std_1, ..., log_std_n
            sz = 2*sz

        return sz

    def sample(self, n=1):
        if self.is_discrete():
            return np.random.randint(self.lo, self.hi+1, (n, ) + self.shape)

        if self.is_continuous():
            return np.random.uniform(self.lo, self.hi+1e-10, (n, ) + self.shape)

    def __repr__(self):
        mid = str(self.shape)
        if self.categorical:
            mid += ", cat: " + str(self.hi)
        return "Space(%s, %s, %s)" % (self.name, mid, str(self.dtype).strip("<class>' "))
