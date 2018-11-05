from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_action(self, obs): ...


from .running import SyncRunningAgent
from .random import RandomAgent
from .a2c import A2CAgent
