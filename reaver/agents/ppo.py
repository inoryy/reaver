import gin.tf
import numpy as np
import tensorflow as tf

from .base.util import tf_run
from . import SyncRunningAgent, ActorCriticAgent


@gin.configurable
class ProximalPolicyOptimizationAgent(SyncRunningAgent, ActorCriticAgent):
    def __init__(
        self,
        sess,
        obs_spec,
        act_spec,
        n_envs=4,
        traj_len=16,
        n_updates=3,
        minibatch_sz=24,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
    ):
        self.n_updates = n_updates
        self.minibatch_sz = minibatch_sz
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        SyncRunningAgent.__init__(self, n_envs)
        ActorCriticAgent.__init__(self, sess, obs_spec, act_spec, n_envs, traj_len)

    def _minimize(self, advantages, returns, train=True):
        inputs = [a.reshape(-1, *a.shape[2:]) for a in self.obs + self.acts]
        tf_inputs = self.model.inputs + self.policy.inputs
        logli_old = tf_run(self.sess, self.policy.logli, tf_inputs, inputs)

        inputs += [advantages.flatten(), returns.flatten(), logli_old]
        tf_inputs += self.loss_inputs

        ops = [self.loss_terms, self.grads_norm]
        if train:
            ops.append(self.train_op)

        loss_terms = grads_norm = None
        for _ in range(self.n_updates):
            idx = np.random.permutation(self.n_envs * self.traj_len)[:self.minibatch_sz]
            minibatch = [inpt[idx] for inpt in inputs]
            loss_terms, grads_norm,  _ = tf_run(self.sess, ops, tf_inputs, minibatch)

        return loss_terms, grads_norm

    def _loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")
        logli_old = tf.placeholder(tf.float32, [None], name="logli_old")

        ratio = tf.exp(self.policy.logli - logli_old)
        clipped_ratio = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)

        policy_loss = -tf.reduce_mean(tf.minimum(adv * ratio, adv * clipped_ratio))
        # TODO clip value loss
        value_loss = tf.reduce_mean((self.value - returns)**2) * self.value_coef
        entropy_loss = tf.reduce_mean(self.policy.entropy) * self.entropy_coef
        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = policy_loss + value_loss - entropy_loss

        return full_loss, [policy_loss, value_loss, entropy_loss], [adv, returns, logli_old]
