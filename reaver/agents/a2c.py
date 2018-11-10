import numpy as np
import tensorflow as tf
from .util.tf import *
from . import SyncRunningAgent, MemoryAgent
from .util import AgentLogger, discounted_cumsum


class A2CAgent(SyncRunningAgent, MemoryAgent):
    def __init__(self, model_cls, obs_spec, act_spec, n_envs=4, batch_sz=16, **kwargs):
        SyncRunningAgent.__init__(self, n_envs)
        MemoryAgent.__init__(self, (batch_sz, n_envs), obs_spec, act_spec)

        self.kwargs = dict(
            lr=0.005,
            discount=0.99,
            gae_lambda=0.95,
            policy_coef=1.0,
            value_coef=0.5,
            entropy_coef=0.001,
            clip_grads_norm=0.0,
            logger_updates=100,
            model_kwargs=dict(),
            optimizer='rmsprop',
        )
        if kwargs:
            self.kwargs.update(kwargs)

        tf.reset_default_graph()
        self.sess = tf.Session()

        opt = optimizers[self.kwargs['optimizer']](self.kwargs['lr'])

        self.model = model_cls(obs_spec, act_spec, **self.kwargs['model_kwargs'])
        self.loss_op, self.loss_terms, self.loss_inputs = self._loss_fn()

        grads, vars = zip(*opt.compute_gradients(self.loss_op))
        if self.kwargs['clip_grads_norm'] > 0.:
            grads, _ = tf.clip_by_global_norm(grads, self.kwargs['clip_grads_norm'])
        self.train_op = opt.apply_gradients(zip(grads, vars))

        self.sess.run(tf.global_variables_initializer())

        self.logger = AgentLogger(self, self.kwargs['logger_updates'])

    def get_action_and_value(self, obs):
        return tf_run(self.sess, [self.model.policy.sample, self.model.value], self.model.inputs, obs)

    def get_action(self, obs):
        return tf_run(self.sess, self.model.policy.sample, self.model.inputs, obs)

    def on_step(self, step, obs, action, reward, done, value=None):
        MemoryAgent.on_step(self, step, obs, action, reward, done, value)
        self.logger.on_step(step)

        if (step + 1) % self.batch_sz > 0:
            return

        next_value = tf_run(self.sess, self.model.value, self.model.inputs, self.next_obs)
        adv, returns = self.compute_advantages_and_returns(next_value)

        inputs = self.obs + self.acts + [adv, returns]
        inputs = [a.reshape(-1, *a.shape[2:]) for a in inputs]
        tf_inputs = self.model.inputs + self.model.policy.inputs + self.loss_inputs

        loss_terms,  _ = tf_run(self.sess, [self.loss_terms, self.train_op], tf_inputs, inputs)

        self.logger.on_update(step, loss_terms, returns, adv, next_value)

    def compute_advantages_and_returns(self, bootstrap_value=0., normalize_adv=True):
        """
        Bootstrap helps with stabilizing advantages with sparse rewards
        GAE can help with reducing variance of policy gradient estimates
        """
        bootstrap_value = np.expand_dims(bootstrap_value, 0)
        values = np.append(self.values, bootstrap_value, axis=0)
        rewards = self.rewards + self.dones * self.kwargs['discount'] * values[1:]
        discounts = self.kwargs['discount'] * (1-self.dones)

        rewards[-1] += (1-self.dones[-1]) * self.kwargs['discount'] * values[-1]
        returns = discounted_cumsum(rewards, discounts)

        if self.kwargs['gae_lambda'] > 0.:
            deltas = self.rewards + self.kwargs['discount'] * values[1:] - values[:-1]
            adv = discounted_cumsum(deltas, self.kwargs['gae_lambda'] * discounts)
        else:
            adv = returns - self.values

        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        return adv, returns

    def _loss_fn(self):
        """
        note: could have calculated advantages directly in TF from returns
        but in future might calculate them differently, e.g. via GAE
        which is not trivial to implement as a tensor ops, so easier to take both in
        """
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")

        policy_loss = tf.reduce_mean(self.model.policy.logli * adv)
        value_loss = tf.losses.mean_squared_error(self.model.value, returns)
        entropy_loss = tf.reduce_mean(self.model.policy.entropy)
        loss_terms = [policy_loss, value_loss, entropy_loss]

        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = self.kwargs['policy_coef']*policy_loss \
            + self.kwargs['value_coef']*value_loss \
            - self.kwargs['entropy_coef']*entropy_loss

        return full_loss, loss_terms + [full_loss], [adv, returns]
