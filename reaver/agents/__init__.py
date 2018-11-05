from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_action(self, obs): ...


from .running import RunningAgent, SyncRunningAgent
from .random import RandomAgent
