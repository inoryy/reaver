from typing import Callable, List, Tuple, Any, Type
from tensorflow.keras import Model
from reaver.envs.base import Spec
from reaver.models.base import MultiPolicy

Done = bool
Reward = int
Action = List[Any]
Observation = List[Any]

PolicyType = Type[MultiPolicy]
ModelBuilder = Callable[[Spec, Spec], Model]
