from abc import ABC, abstractmethod
from .env_spec import Space, Spec


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


from .sc2_env import SC2Env
from .mp_env import MultiProcEnv
