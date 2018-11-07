from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_action(self, obs): ...


from .running import SyncRunningAgent, RunningAgent
from .memory import MemoryAgent
from .random import RandomAgent
from .a2c import A2CAgent
