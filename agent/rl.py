import numpy as np

from pysc2.lib import actions
from pysc2.agents import base_agent

from model import simple
from common import n_channels, preprocess_inputs


class RLAgent(base_agent.BaseAgent):
    def __init__(self, feats):
        self.feats = feats
        self.model = simple(*n_channels(feats))
        self.model.compile(optimizer='adam', loss='mse')

    def step(self, obs):
        x = preprocess_inputs(obs, self.feats)
        (screen_cat_x, screen_num_x), (minimap_cat_x, minimap_num_x) = x
        spatial_action_dists, values = self.model.predict([screen_cat_x, screen_num_x, minimap_cat_x, minimap_num_x])
        acts = []
        for i in range(len(obs)):
            if 12 not in obs[i].observation["available_actions"]:
                acts.append(actions.FunctionCall(7, [[0]]))
                continue
            spatial_action = np.random.choice(64 * 64, p=spatial_action_dists[i])
            coords = np.unravel_index(spatial_action, (64, 64))
            acts.append(actions.FunctionCall(12, [[1], coords]))
        return acts