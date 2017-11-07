import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from pysc2.lib import actions
from pysc2.agents.base_agent import BaseAgent

from model import fully_conv
from common.util import RolloutStorage
from common import n_channels, preprocess_inputs, unravel_coords


def sample(policy):
    # based on https://github.com/pekaalto/sc2aibot/blob/master/common/util.py#L5-L11
    # TODO verify it really does what's expected
    u = tf.random_uniform(tf.shape(policy))
    return tf.argmax(tf.log(u) / policy, axis=1, output_type=tf.int32)


def select(policy, acts):
    return tf.gather_nd(policy, tf.stack([tf.range(tf.shape(policy)[0]), acts], axis=1))


class RLAgent(BaseAgent):
    def __init__(self, sess, feats, n_steps=40):
        self.sess = sess
        self.feats = feats
        self.n_steps = n_steps
        self.rollouts = RolloutStorage()
        self.inputs, (self.spatial_policy, self.value) = fully_conv(*n_channels(feats))
        self.spatial_action = sample(self.spatial_policy)
        loss_fn, self.loss_inputs = self.loss_func()
        self.train_op = layers.optimize_loss(loss=loss_fn, optimizer=tf.train.AdamOptimizer(), learning_rate=None,
                                             global_step=tf.train.get_global_step(), clip_gradients=500.)
        self.sess.run(tf.global_variables_initializer())
        self.rewards = []

    def step(self, obs):
        reward = [ob.reward for ob in obs]
        self.rewards.append(reward)
        if obs[0].first():
            print(np.mean(np.sum(self.rewards, axis=0)))
            self.rewards = []

        x = preprocess_inputs(obs, self.feats)

        if self.steps > 0 and self.steps % self.n_steps == 0:
            self.rollouts.rewards.append(reward)
            last_value = self.sess.run(self.value, feed_dict=dict(zip(self.inputs, x)))
            self.rollouts.compute_returns(last_value, 0.99)
            self.train()
            self.rollouts = RolloutStorage()
        self.steps += 1

        spatial_action, value = self.sess.run([self.spatial_action, self.value], feed_dict=dict(zip(self.inputs, x)))
        self.rollouts.insert(x, spatial_action, reward, value)

        coords = unravel_coords(spatial_action)
        acts = []
        for i in range(len(obs)):
            if 12 not in obs[i].observation["available_actions"]:
                acts.append(actions.FunctionCall(7, [[0]]))
                continue
            # https://github.com/deepmind/pysc2/issues/103
            y, x = coords[i]
            acts.append(actions.FunctionCall(12, [[0], (x, y)]))
        return acts

    def loss_func(self):
        returns = tf.placeholder(tf.float32, [None])
        adv = tf.stop_gradient(returns - self.value)
        logli = select(tf.log(self.spatial_policy), self.spatial_action)
        entropy = tf.reduce_sum(self.spatial_policy * tf.log(self.spatial_policy), axis=1)

        policy_loss = -tf.reduce_mean(logli * adv)
        value_loss = tf.reduce_mean(tf.pow(adv, 2))
        entropy_loss = 1e-3 * tf.reduce_mean(entropy)

        return policy_loss + value_loss + entropy_loss, [returns]

    def train(self):
        self.sess.run(self.train_op, dict(zip(self.inputs + self.loss_inputs, self.rollouts.inputs())))
