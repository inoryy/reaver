import numpy as np
from pysc2.lib import actions
from pysc2.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def step(self, obs):
        acts = []
        for i in range(len(obs)):
            fid = np.random.choice(obs[i].observation["available_actions"])
            args = [[np.random.randint(0, size) for size in arg.sizes] for arg in self.action_spec.functions[fid].args]
            acts.append(actions.FunctionCall(fid, args))
        return acts
