import tensorflow as tf

from pysc2.lib import actions
from pysc2.agents.base_agent import BaseAgent

from model import fully_conv
from common import n_channels, preprocess_inputs, unravel_coords


class RLAgent(BaseAgent):
    def __init__(self, sess, feats):
        self.sess = sess
        self.feats = feats
        self.inputs, (self.spatial_policy, self.spatial_action, self.value) = fully_conv(*n_channels(feats))
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs):
        x = preprocess_inputs(obs, self.feats)
        spatial_actions = unravel_coords(self.sess.run(self.spatial_action, feed_dict=dict(zip(self.inputs, x))))

        acts = []
        for i in range(len(obs)):
            if 12 not in obs[i].observation["available_actions"]:
                acts.append(actions.FunctionCall(7, [[0]]))
                continue
            acts.append(actions.FunctionCall(12, [[1], spatial_actions[i]]))
        return acts
