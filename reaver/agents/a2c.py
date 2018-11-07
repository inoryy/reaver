import numpy as np
import tensorflow as tf
from . import SyncRunningAgent, MemoryAgent
from .util import AgentLogger


class A2CAgent(SyncRunningAgent, MemoryAgent):
    def __init__(self, model_cls, obs_spec, act_spec, n_envs=1, batch_sz=8, **kwargs):
        SyncRunningAgent.__init__(self, n_envs)
        MemoryAgent.__init__(self, (batch_sz, n_envs), obs_spec, act_spec)

        self.coefs = dict(
            lr=0.001,
            policy=1.0,
            value=1.0,
            entropy=0.0,
            discount=0.99,)
        if kwargs:
            self.coefs.update(kwargs)

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.model = model_cls(obs_spec, act_spec)
        self.loss_op, self.loss_terms, self.loss_inputs = self._loss_fn()
        self.train_op = tf.train.AdamOptimizer(self.coefs['lr']).minimize(self.loss_op)

        self.sess.run(tf.global_variables_initializer())

        self.logger = AgentLogger(self)

    def get_action_and_value(self, obs):
        feed_dict = dict(zip(self.model.inputs, obs))
        return self.sess.run([self.model.policy.sample, self.model.value], feed_dict=feed_dict)

    def get_action(self, obs):
        feed_dict = dict(zip(self.model.inputs, obs))
        return self.sess.run(self.model.policy.sample, feed_dict=feed_dict)

    def on_step(self, step, obs, action, reward, done, value=None):
        MemoryAgent.on_step(self, step, obs, action, reward, done, value)
        if (step + 1) % self.batch_sz > 0:
            return

        next_value = self.tf_run(self.model.value, self.model.inputs, self.next_obs)
        adv = self.compute_advantages(next_value)

        inputs = self.obs + self.acts + [adv]
        inputs = [a.reshape(-1, *a.shape[2:]) for a in inputs]
        tf_inputs = self.model.inputs + self.model.policy.action_inputs + self.loss_inputs

        loss_terms,  _ = self.tf_run([self.loss_terms, self.train_op], tf_inputs, inputs)

        self.logger.on_step(step, loss_terms, adv, next_value)

    def compute_advantages(self, next_value, normalize=False):
        returns = np.zeros((self.batch_sz+1, self.n_envs), dtype=np.float32)
        returns[-1] = next_value

        for t in range(self.batch_sz-1, -1, -1):
            returns[t] = self.rewards[t] + self.coefs['discount'] * returns[t+1] * (1-self.dones[t])
        adv = returns[:-1] - self.values

        if normalize:
            adv = (adv - np.mean(adv, axis=0)) / (np.std(adv, axis=0) + 1e-12)

        return adv

    def _loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")

        policy = tf.reduce_mean(self.model.policy.logli * adv)
        value = tf.reduce_mean(tf.square(adv))
        entropy = tf.reduce_mean(self.model.policy.entropy)
        loss_terms = [policy, value, entropy]

        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = self.coefs['policy']*policy + self.coefs['value']*value - self.coefs['entropy']*entropy

        return full_loss, loss_terms, [adv]

    def tf_run(self, tf_op, tf_inputs, inputs):
        return self.sess.run(tf_op, feed_dict=dict(zip(tf_inputs, inputs)))
