from pysc2.agents import base_agent
from common import preprocess_inputs


class RLAgent(base_agent.BaseAgent):
    def __init__(self, model, algorithm, feats):
        self.feats = feats
        self.model = model
        self.algo = algorithm

    def step(self, obs):
        x = preprocess_inputs(obs, self.feats)
        spatial_action, value = self.model.forward(x)
