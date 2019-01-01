from .base import *
from .random import RandomAgent
from .a2c import AdvantageActorCriticAgent
from .ppo import ProximalPolicyOptimizationAgent

A2C = AdvantageActorCriticAgent
PPO = ProximalPolicyOptimizationAgent

registry = {
    'a2c': A2C,
    'ppo': PPO
}
