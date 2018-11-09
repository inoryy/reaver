import numpy as np
import tensorflow as tf
from . import SyncRunningAgent, MemoryAgent
from .util import AgentLogger


class A2CAgent(SyncRunningAgent, MemoryAgent):
    def __init__(self, model_cls, obs_spec, act_spec, n_envs=4, batch_sz=16, clip_grads_norm=0.0, **kwargs):
        SyncRunningAgent.__init__(self, n_envs)
        MemoryAgent.__init__(self, (batch_sz, n_envs), obs_spec, act_spec)

        self.coefs = dict(
            lr=0.005,
            policy=1.0,
            value=0.5,
            entropy=0.001,
            discount=0.99,
        )
        if kwargs:
            self.coefs.update(kwargs)

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.model = model_cls(obs_spec, act_spec)

        self.loss_op, self.loss_terms, self.loss_inputs = self._loss_fn()
        opt = tf.train.RMSPropOptimizer(self.coefs['lr'])
        grads, vars = zip(*opt.compute_gradients(self.loss_op))
        if clip_grads_norm > 0.:
            grads, _ = tf.clip_by_global_norm(grads, clip_grads_norm)
        self.train_op = opt.apply_gradients(zip(grads, vars))

        self.sess.run(tf.global_variables_initializer())

        self.logger = AgentLogger(self)

    def get_action_and_value(self, obs):
        feed_dict = dict(zip(self.model.inputs, obs))
        return self.sess.run([self.model.policy.sample, self.model.value], feed_dict=feed_dict)

    def get_action(self, obs):
        feed_dict = dict(zip(self.model.inputs, obs))
        return self.sess.run(self.model.policy.sample, feed_dict=feed_dict)

    def run(self, env, n_steps=1000000):
        SyncRunningAgent.run(self, env, n_steps*self.batch_sz)

    def on_step(self, step, obs, action, reward, done, value=None):
        MemoryAgent.on_step(self, step, obs, action, reward, done, value)
        if (step + 1) % self.batch_sz > 0:
            return

        next_value = self.tf_run(self.model.value, self.model.inputs, self.next_obs)
        adv, returns = self.compute_advantages_and_returns(next_value)

        inputs = self.obs + self.acts + [adv, returns]
        inputs = [a.reshape(-1, *a.shape[2:]) for a in inputs]
        tf_inputs = self.model.inputs + self.model.policy.action_inputs + self.loss_inputs

        loss_terms,  _ = self.tf_run([self.loss_terms, self.train_op], tf_inputs, inputs)

        self.logger.on_step(step, loss_terms, returns, adv, next_value)

    def compute_advantages_and_returns(self, bootstrap_value=0., normalize_returns=False, normalize_adv=False):
        """
        Bootstrap helps with stabilizing advantages with sparse rewards
        Returns normalization can help with stabilizing value loss
        Advantage normalization can help with stabilizing policy loss, but can lead to large swings if rewards are sparse
        """
        returns = np.zeros((self.batch_sz+1, self.n_envs), dtype=np.float32)
        values = np.zeros((self.batch_sz+1, self.n_envs), dtype=np.float32)
        values[:-1] = self.values
        returns[-1] = values[-1] = bootstrap_value

        for t in range(self.batch_sz-1, -1, -1):
            returns[t] = self.rewards[t] + self.coefs['discount'] * returns[t+1] * (1-self.dones[t])
            # avoid killing bootstrap signal in terminal states
            returns[t] += self.coefs['discount'] * values[t+1] * self.dones[t]
        returns = returns[:-1]

        if normalize_returns:
            r_mu, r_std = np.mean(returns, axis=0), np.std(returns, axis=0) + 1e-12

            returns = (returns - r_mu) / r_std
            # have to re-scale baseline for advantages
            self.values = self.values * r_std + r_mu

        adv = returns - self.values

        if normalize_adv:
            adv = (adv - np.mean(adv, axis=0)) / (np.std(adv, axis=0) + 1e-12)

        return adv, returns

    def _loss_fn(self):
        """
        note: could have calculated advantages directly in TF from returns
        but in future might calculate them differently, e.g. via GAE
        which is not trivial to implement as a tensor ops, so easier to take both in
        """
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")

        policy = tf.reduce_mean(self.model.policy.logli * adv)
        value = tf.losses.mean_squared_error(self.model.value, returns)
        entropy = tf.reduce_mean(self.model.policy.entropy)
        loss_terms = [policy, value, entropy]

        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = self.coefs['policy']*policy + self.coefs['value']*value - self.coefs['entropy']*entropy

        return full_loss, loss_terms + [full_loss], [adv, returns]

    def tf_run(self, tf_op, tf_inputs, inputs):
        return self.sess.run(tf_op, feed_dict=dict(zip(tf_inputs, inputs)))
