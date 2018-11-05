from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def start(self): ...

    @abstractmethod
    def step(self, action): ...

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def stop(self): ...

    @abstractmethod
    def obs_spec(self): ...

    @abstractmethod
    def act_spec(self): ...


class Space:
    def __init__(self, shape=(1,), dtype=float, domain=(0, 1), categorical=False):
        self.lo, self.hi = domain
        self.categorical = categorical
        self.shape, self.dtype = shape, dtype

    def __repr__(self):
        return "Space(sh: %s, dt: %s, dom: (%d, %d)" % (str(self.shape), str(self.dtype), self.lo, self.hi)


class Spec:
    def __init__(self, *spaces):
        self.spaces = spaces

    def __repr__(self):
        return "Spec:" + str(self.spaces)
