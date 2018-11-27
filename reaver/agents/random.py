import numpy as np
from reaver.agents.base import SyncRunningAgent


class RandomAgent(SyncRunningAgent):
    def __init__(self, act_spec, n_envs):
        super().__init__(n_envs)
        self.act_spec = act_spec

    def get_action(self, obs):
        function_id = [np.random.choice(np.argwhere(obs[2][i] > 0).flatten()) for i in range(self.n_envs)]
        args = [[[np.random.randint(0, size) for size in arg.shape] for _ in range(self.n_envs)]
                for arg in self.act_spec.spaces[1:]]
        return [function_id] + args
