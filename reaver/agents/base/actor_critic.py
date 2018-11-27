import gin.tf
import numpy as np
import tensorflow as tf
from abc import abstractmethod

from reaver.utils import Logger
from reaver.envs.base import Spec
from reaver.agents.base import MemoryAgent
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType


class ActorCriticAgent(MemoryAgent):
    """
    Abstract class, unifies deep actor critic functionality
    Handles on_step callbacks, either updating current batch
    or executing one training step if the batch is ready

    Extending classes only need to implement loss_fn method
    """
    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        model_fn: ModelBuilder,
        policy_cls: PolicyType,
        sess_mgr: SessionManager = None,
        traj_len=16,
        batch_sz=16,
        discount=0.99,
        gae_lambda=0.95,
        clip_rewards=0.0,
        normalize_advantages=True,
        bootstrap_terminals=False,
        clip_grads_norm=0.0,
        optimizer=tf.train.AdamOptimizer(),
        logger=Logger()
    ):
        MemoryAgent.__init__(self, obs_spec, act_spec, traj_len, batch_sz)

        if not sess_mgr:
            sess_mgr = SessionManager()

        self.sess_mgr = sess_mgr
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_rewards = clip_rewards
        self.normalize_advantages = normalize_advantages
        self.bootstrap_terminals = bootstrap_terminals
        self.logger = logger

        self.model = model_fn(obs_spec, act_spec)
        self.value = self.model.outputs[-1]
        self.policy = policy_cls(act_spec, self.model.outputs[:-1])
        self.loss_op, self.loss_terms, self.loss_inputs = self.loss_fn()

        grads, vars = zip(*optimizer.compute_gradients(self.loss_op))
        self.grads_norm = tf.global_norm(grads)
        if clip_grads_norm > 0.:
            grads, _ = tf.clip_by_global_norm(grads, clip_grads_norm, self.grads_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, vars), global_step=self.sess_mgr.global_step)

        self.sess_mgr.restore_or_init()
        # NB! changing trajectory length in-between checkpoints will break the logs
        self.n_batches = self.sess_mgr.start_step
        self.start_step = self.sess_mgr.start_step * traj_len

    def get_action_and_value(self, obs):
        return self.sess_mgr.run([self.policy.sample, self.value], self.model.inputs, obs)

    def get_action(self, obs):
        return self.sess_mgr.run(self.policy.sample, self.model.inputs, obs)

    def on_step(self, step, obs, action, reward, done, value=None):
        MemoryAgent.on_step(self, step, obs, action, reward, done, value)
        self.logger.on_step(step, reward, done)

        if not self.batch_ready():
            return

        next_values = self.sess_mgr.run(self.value, self.model.inputs, self.last_obs)
        adv, returns = self.compute_advantages_and_returns(next_values)

        loss_terms, grads_norm = self.minimize(adv, returns)

        self.sess_mgr.on_update(self.n_batches)
        self.logger.on_update(self.n_batches, loss_terms, grads_norm, returns, adv, next_values)

    def minimize(self, advantages, returns):
        inputs = self.obs + self.acts + [advantages, returns]
        inputs = [a.reshape(-1, *a.shape[2:]) for a in inputs]
        tf_inputs = self.model.inputs + self.policy.inputs + self.loss_inputs

        ops = [self.loss_terms, self.grads_norm]
        if self.sess_mgr.training_enabled:
            ops.append(self.train_op)

        loss_terms, grads_norm, *_ = self.sess_mgr.run(ops, tf_inputs, inputs)
        return loss_terms, grads_norm

    def compute_advantages_and_returns(self, bootstrap_value):
        """
        Bootstrap helps with stabilizing advantages with sparse rewards
        GAE can help with reducing variance of policy gradient estimates
        """
        bootstrap_value = np.expand_dims(bootstrap_value, 0)
        values = np.append(self.values, bootstrap_value, axis=0)
        rewards = self.rewards.copy()

        if self.clip_rewards > 0.0:
            np.clip(rewards, -self.clip_rewards, self.clip_rewards, out=rewards)

        if self.bootstrap_terminals:
            rewards += self.dones * self.discount * values[:-1]
        discounts = self.discount * (1-self.dones)

        rewards[-1] += (1-self.dones[-1]) * self.discount * values[-1]
        returns = self.discounted_cumsum(rewards, discounts)

        if self.gae_lambda > 0.:
            deltas = self.rewards + discounts * values[1:] - values[:-1]
            if self.bootstrap_terminals:
                deltas += self.dones * self.discount * values[:-1]
            adv = self.discounted_cumsum(deltas, self.gae_lambda * discounts)
        else:
            adv = returns - self.values

        if self.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        return adv, returns

    def on_start(self):
        self.logger.on_start()

    def on_finish(self):
        self.logger.on_finish()

    @staticmethod
    def discounted_cumsum(x, discount):
        y = np.zeros_like(x)
        y[-1] = x[-1]
        for t in range(x.shape[0] - 2, -1, -1):
            y[t] = x[t] + discount[t] * y[t + 1]
        return y

    @abstractmethod
    def loss_fn(self): ...
