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


from .spec import Space, Spec
from .multiproc import MultiProcEnv
