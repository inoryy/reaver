from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_action(self, obs): ...


from .memory import MemoryAgent
from .running import SyncRunningAgent
from .actor_critic import ActorCriticAgent, DEFAULTS
