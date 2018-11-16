import gin.tf
import numpy as np
import tensorflow as tf
from abc import abstractmethod

from reaver.models import mlp, MultiPolicy
from reaver.agents.base import MemoryAgent
from .util import tf_run, discounted_cumsum, AgentLogger


@gin.configurable
class ActorCriticAgent(MemoryAgent):
    def __init__(
        self,
        sess,
        obs_spec,
        act_spec,
        n_envs=4,
        traj_len=16,
        network_fn=mlp,
        policy_cls=MultiPolicy,
        discount=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        bootstrap_terminals=False,
        clip_grads_norm=0.0,
        optimizer=tf.train.AdamOptimizer()
    ):
        MemoryAgent.__init__(self, obs_spec, act_spec, (traj_len, n_envs))

        self.sess = sess
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.bootstrap_terminals = bootstrap_terminals

        self.network = network_fn(obs_spec, act_spec)
        self.value = self.network.outputs[-1]
        self.policy = policy_cls(act_spec, self.network.outputs[:-1])
        self.loss_op, self.loss_terms, self.loss_inputs = self._loss_fn()

        grads, vars = zip(*optimizer.compute_gradients(self.loss_op))
        self.grads_norm = tf.linalg.global_norm(grads)
        if clip_grads_norm > 0.:
            grads, _ = tf.clip_by_global_norm(grads, clip_grads_norm, self.grads_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, vars))

        self.sess.run(tf.global_variables_initializer())

        self.logger = AgentLogger(self)

    def get_action_and_value(self, obs):
        return tf_run(self.sess, [self.policy.sample, self.value], self.network.inputs, obs)

    def get_action(self, obs):
        return tf_run(self.sess, self.policy.sample, self.network.inputs, obs)

    def on_step(self, step, obs, action, reward, done, value=None):
        MemoryAgent.on_step(self, step, obs, action, reward, done, value)
        self.logger.on_step(step)

        if (step + 1) % self.traj_len > 0:
            return

        next_value = tf_run(self.sess, self.value, self.network.inputs, self.next_obs)
        adv, returns = self.compute_advantages_and_returns(next_value)

        loss_terms, grads_norm = self._minimize(adv, returns)

        self.logger.on_update(step, loss_terms, grads_norm, returns, adv, next_value)

    def compute_advantages_and_returns(self, bootstrap_value=0.):
        """
        Bootstrap helps with stabilizing advantages with sparse rewards
        GAE can help with reducing variance of policy gradient estimates
        """
        bootstrap_value = np.expand_dims(bootstrap_value, 0)
        values = np.append(self.values, bootstrap_value, axis=0)
        rewards = self.rewards.copy()
        if self.bootstrap_terminals:
            rewards += self.dones * self.discount * values[:-1]
        discounts = self.discount * (1-self.dones)

        rewards[-1] += (1-self.dones[-1]) * self.discount * values[-1]
        returns = discounted_cumsum(rewards, discounts)

        if self.gae_lambda > 0.:
            deltas = self.rewards + discounts * values[1:] - values[:-1]
            if self.bootstrap_terminals:
                deltas += self.dones * self.discount * values[:-1]
            adv = discounted_cumsum(deltas, self.gae_lambda * discounts)
        else:
            adv = returns - self.values

        if self.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        return adv, returns

    def _minimize(self, advantages, returns, train=True):
        inputs = self.obs + self.acts + [advantages, returns]
        inputs = [a.reshape(-1, *a.shape[2:]) for a in inputs]
        tf_inputs = self.network.inputs + self.policy.inputs + self.loss_inputs

        ops = [self.loss_terms, self.grads_norm]
        if train:
            ops.append(self.train_op)

        loss_terms, grads_norm, *_ = tf_run(self.sess, ops, tf_inputs, inputs)
        return loss_terms, grads_norm

    @abstractmethod
    def _loss_fn(self): ...
