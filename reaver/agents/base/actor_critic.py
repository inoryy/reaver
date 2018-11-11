from abc import abstractmethod
import numpy as np
import tensorflow as tf
from .memory import MemoryAgent
from .util import tf_run, discounted_cumsum, AgentLogger

optimizers = dict(
    adam=tf.train.AdamOptimizer,
    rmsprop=tf.train.RMSPropOptimizer
)


class ActorCriticAgent(MemoryAgent):
    def __init__(self, model_cls, obs_spec, act_spec, base_shape, **kwargs):
        MemoryAgent.__init__(self, base_shape, obs_spec, act_spec)

        self.kwargs = dict(
            lr=0.005,
            optimizer='adam',
            clip_grads_norm=0.0,
            discount=0.99,
            gae_lambda=0.95,
            normalize_advantages=True,
            bootstrap_terminals=False,
            model_kwargs=dict(),
            logger_kwargs=dict(),
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

        self.logger = AgentLogger(self, **self.kwargs['logger_kwargs'])

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

        loss_terms = self._train(adv, returns)

        self.logger.on_update(step, loss_terms, returns, adv, next_value)

    def compute_advantages_and_returns(self, bootstrap_value=0.):
        """
        Bootstrap helps with stabilizing advantages with sparse rewards
        GAE can help with reducing variance of policy gradient estimates
        """
        bootstrap_value = np.expand_dims(bootstrap_value, 0)
        values = np.append(self.values, bootstrap_value, axis=0)
        rewards = self.rewards.copy()
        if self.kwargs['bootstrap_terminals']:
            rewards += self.dones * self.kwargs['discount'] * values[1:]
        discounts = self.kwargs['discount'] * (1-self.dones)

        rewards[-1] += (1-self.dones[-1]) * self.kwargs['discount'] * values[-1]
        returns = discounted_cumsum(rewards, discounts)

        if self.kwargs['gae_lambda'] > 0.:
            deltas = self.rewards + discounts * values[1:] - values[:-1]
            if self.kwargs['bootstrap_terminals']:
                deltas += self.dones * self.kwargs['discount'] * values[1:]
            adv = discounted_cumsum(deltas, self.kwargs['gae_lambda'] * discounts)
        else:
            adv = returns - self.values

        if self.kwargs['normalize_advantages']:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        return adv, returns

    def _train(self, advantages, returns):
        inputs = self.obs + self.acts + [advantages, returns]
        inputs = [a.reshape(-1, *a.shape[2:]) for a in inputs]
        tf_inputs = self.model.inputs + self.model.policy.inputs + self.loss_inputs

        loss_terms,  _ = tf_run(self.sess, [self.loss_terms, self.train_op], tf_inputs, inputs)
        return loss_terms

    @abstractmethod
    def _loss_fn(self): ...
