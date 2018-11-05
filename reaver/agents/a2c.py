import numpy as np
import tensorflow as tf
from . import SyncRunningAgent


class A2CAgent(SyncRunningAgent):
    def __init__(self, sess, model, n_envs, batch_sz=16, discount=0.99):
        super().__init__(n_envs)
        self.sess, self.policy = sess, model
        self.batch_sz, self.discount = batch_sz, discount

    def get_action(self, obs):
        feed_dict = dict(zip(self.policy.inputs, obs))
        return self.sess.run(self.policy.sample, feed_dict=feed_dict)

    def compute_returns(self, rewards, dones, last_value):
        returns = np.zeros((self.batch_sz+1, self.n_envs), dtype=np.float32)
        returns[-1] = last_value
        for t in range(self.batch_sz-1, -1, -1):
            returns[t] = rewards[t] + self.discount * returns[t+1] * (1-dones[t])
        return returns[:-1]
