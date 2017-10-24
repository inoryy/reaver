from pysc2.agents import base_agent


class RLAgent(base_agent.BaseAgent):
    def __init__(self, model, algorithm):
        self.model = model
        self.algo = algorithm

    def step(self, obs):
        pass