from .base import Env, Space, Spec
from .sc2 import SC2Env

import importlib
if importlib.util.find_spec("gym") is not None:
    from .gym import GymEnv
