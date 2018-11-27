from abc import ABC, abstractmethod
from reaver.utils.typing import *
from .spec import Spec


class Env(ABC):
    """
    Abstract Base Class for all environments supported by Reaver
    Acts as a glue between the agents, models and envs modules
    Implementing class can be a simple wrapper (e.g. over openAI Gym)

    Note: observation / action specs contain a list of spaces,
          this is implicitly assumed across all Reaver components
    """
    def __init__(self, _id: str, render=False, reset_done=True, max_ep_len=None):
        self.id = _id
        self.render = render
        self.reset_done = reset_done
        self.max_ep_len = max_ep_len if max_ep_len else float('inf')

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, Reward, Done]: ...

    @abstractmethod
    def reset(self) -> Observation: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def obs_spec(self) -> Spec: ...

    @abstractmethod
    def act_spec(self) -> Spec: ...
