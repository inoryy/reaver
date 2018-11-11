import numpy as np
import tensorflow as tf
from .base.util import tf_run
from . import SyncRunningAgent, ActorCriticAgent


class ProximalPolicyOptimizationAgent(SyncRunningAgent, ActorCriticAgent):
    def __init__(self, model_cls, obs_spec, act_spec, n_envs=4, batch_sz=16, **kwargs):
        _kwargs = dict(
            lr=0.001,
            policy_coef=1.0,
            value_coef=0.005,
            entropy_coef=0.01,
            clip_ratio=0.2,
            ppo_updates=3,
            minibatch_sz=n_envs * batch_sz // 2
        )
        if kwargs:
            _kwargs.update(kwargs)

        SyncRunningAgent.__init__(self, n_envs)
        ActorCriticAgent.__init__(self, model_cls, obs_spec, act_spec, (batch_sz, n_envs), **_kwargs)

    def _train(self, advantages, returns):
        tf_inputs = self.model.inputs + self.model.policy.inputs
        inputs = [a.reshape(-1, *a.shape[2:]) for a in self.obs + self.acts]
        logli_old = tf_run(self.sess, self.model.policy.logli, tf_inputs, inputs)

        tf_inputs += self.loss_inputs
        inputs += [advantages.flatten(), returns.flatten(), logli_old]

        loss_terms = [-1]*4
        for _ in range(self.kwargs['ppo_updates']):
            idx = np.random.randint(0, inputs[0].shape[0], self.kwargs['minibatch_sz'])
            minibatch = [inpt[idx] for inpt in inputs]
            loss_terms,  _ = tf_run(self.sess, [self.loss_terms, self.train_op], tf_inputs, minibatch)
        return loss_terms

    def _loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")
        logli_old = tf.placeholder(tf.float32, [None], name="logli_old")

        policy = self.model.policy
        eps = self.kwargs['clip_ratio']

        ratio = tf.exp(policy.logli - logli_old)
        policy_loss = tf.reduce_mean(tf.minimum(adv * ratio, adv * tf.clip_by_value(ratio, 1-eps, 1+eps)))
        value_loss = tf.reduce_mean((self.model.value - returns) ** 2)
        entropy_loss = tf.reduce_mean(self.model.policy.entropy)
        loss_terms = [policy_loss, value_loss, entropy_loss]

        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = self.kwargs['policy_coef']*policy_loss \
            + self.kwargs['value_coef']*value_loss \
            - self.kwargs['entropy_coef']*entropy_loss

        return full_loss, loss_terms + [full_loss], [adv, returns, logli_old]
