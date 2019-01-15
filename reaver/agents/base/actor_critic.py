import gin.tf
import numpy as np
import tensorflow as tf
from abc import abstractmethod

from reaver.envs.base import Spec
from reaver.agents.base import MemoryAgent
from reaver.utils import Logger
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType

DEFAULTS = dict(
    model_fn=None,
    policy_cls=None,
    optimizer=None,
    learning_rate=0.0003,
    value_coef=0.5,
    entropy_coef=0.01,
    traj_len=16,
    batch_sz=16,
    discount=0.99,
    gae_lambda=0.95,
    clip_rewards=0.0,
    clip_grads_norm=0.0,
    normalize_returns=False,
    normalize_advantages=False,
)


@gin.configurable('ACAgent')
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
        model_fn: ModelBuilder=None,
        policy_cls: PolicyType=None,
        sess_mgr: SessionManager=None,
        optimizer: tf.train.Optimizer=None,
        value_coef=DEFAULTS['value_coef'],
        entropy_coef=DEFAULTS['entropy_coef'],
        traj_len=DEFAULTS['traj_len'],
        batch_sz=DEFAULTS['batch_sz'],
        discount=DEFAULTS['discount'],
        gae_lambda=DEFAULTS['gae_lambda'],
        clip_rewards=DEFAULTS['clip_rewards'],
        clip_grads_norm=DEFAULTS['clip_grads_norm'],
        normalize_returns=DEFAULTS['normalize_returns'],
        normalize_advantages=DEFAULTS['normalize_advantages'],
    ):
        MemoryAgent.__init__(self, obs_spec, act_spec, traj_len, batch_sz)

        if not sess_mgr:
            sess_mgr = SessionManager()

        if not optimizer:
            optimizer = tf.train.AdamOptimizer(learning_rate=DEFAULTS['learning_rate'])

        self.sess_mgr = sess_mgr
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_rewards = clip_rewards
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages

        self.model = model_fn(obs_spec, act_spec)
        self.value = self.model.outputs[-1]
        self.policy = policy_cls(act_spec, self.model.outputs[:-1])
        self.loss_op, self.loss_terms, self.loss_inputs = self.loss_fn()

        grads, vars = zip(*optimizer.compute_gradients(self.loss_op))
        self.grads_norm = tf.global_norm(grads)
        if clip_grads_norm > 0.:
            grads, _ = tf.clip_by_global_norm(grads, clip_grads_norm, self.grads_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, vars), global_step=sess_mgr.global_step)
        self.minimize_ops = self.make_minimize_ops()

        sess_mgr.restore_or_init()
        self.n_batches = sess_mgr.start_step
        self.start_step = sess_mgr.start_step * traj_len

        self.logger = Logger()

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

        loss_terms, grads_norm, *_ = self.sess_mgr.run(self.minimize_ops, tf_inputs, inputs)

        return loss_terms, grads_norm

    def compute_advantages_and_returns(self, bootstrap_value):
        """
        GAE can help with reducing variance of policy gradient estimates
        """
        if self.clip_rewards > 0.0:
            np.clip(self.rewards, -self.clip_rewards, self.clip_rewards, out=self.rewards)

        rewards = self.rewards.copy()
        rewards[-1] += (1-self.dones[-1]) * self.discount * bootstrap_value

        masked_discounts = self.discount * (1-self.dones)

        returns = self.discounted_cumsum(rewards, masked_discounts)

        if self.gae_lambda > 0.:
            values = np.append(self.values, np.expand_dims(bootstrap_value, 0), axis=0)
            # d_t = r_t + g * V(s_{t+1}) - V(s_t)
            deltas = self.rewards + masked_discounts * values[1:] - values[:-1]
            adv = self.discounted_cumsum(deltas, self.gae_lambda * masked_discounts)
        else:
            adv = returns - self.values

        if self.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        if self.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)

        return adv, returns

    def on_start(self):
        self.logger.on_start()

    def on_finish(self):
        self.logger.on_finish()

    def make_minimize_ops(self):
        ops = [self.loss_terms, self.grads_norm]
        if self.sess_mgr.training_enabled:
            ops.append(self.train_op)
        # appending extra model update ops (e.g. running stats)
        # note: this will most likely break if model.compile() is used
        ops.extend(self.model.get_updates_for(None))
        return ops

    @staticmethod
    def discounted_cumsum(x, discount):
        y = np.zeros_like(x)
        y[-1] = x[-1]
        for t in range(x.shape[0]-2, -1, -1):
            y[t] = x[t] + discount[t] * y[t+1]
        return y

    @abstractmethod
    def loss_fn(self): ...
