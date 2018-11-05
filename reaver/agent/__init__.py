from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, obs_spec, act_spec):
        self.obs_spec, self.act_spec = obs_spec, act_spec

    @abstractmethod
    def get_action(self, obs): ...


from .running_agent import RunningAgent, SyncRunningAgent
from .random_agent import RandomAgent
