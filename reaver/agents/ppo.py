import gin.tf
import numpy as np
import tensorflow as tf

from reaver.envs.base import Spec
from reaver.utils import StreamLogger
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType
from reaver.agents.base import SyncRunningAgent, ActorCriticAgent, DEFAULTS


@gin.configurable('PPOAgent')
class ProximalPolicyOptimizationAgent(SyncRunningAgent, ActorCriticAgent):
    """
    PPO: clipped version of the Proximal Policy Optimization algorithm

    Here "clipped" refers to how trusted policy region is enforced.
    While orig. PPO relied on KL divergence, this clips the pi / pi_old ratio.

    See article for more details: https://arxiv.org/abs/1707.06347

    PPO specific parameters:

    :param n_epochs: number of times optimizer goes through full batch_sz*traj_len set
    :param minibatch_sz: size of the randomly sampled minibatch
    :param clip_ratio: max interval for pi / pi_old: [1-clip_ratio, 1+clip_ratio]
    :param clip_value: max interval for new value error: [old_value-clip_value, old_value+clip_value]
    """
    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        model_fn: ModelBuilder=None,
        policy_cls: PolicyType=None,
        sess_mgr: SessionManager=None,
        optimizer: tf.train.Optimizer=None,
        n_envs=4,
        n_epochs=3,
        minibatch_sz=128,
        clip_ratio=0.2,
        clip_value=0.5,
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
        kwargs = {k: v for k, v in locals().items() if k in DEFAULTS and DEFAULTS[k] != v}

        self.n_epochs = n_epochs
        self.minibatch_sz = minibatch_sz
        self.clip_ratio = clip_ratio
        self.clip_value = clip_value

        SyncRunningAgent.__init__(self, n_envs)
        ActorCriticAgent.__init__(self, obs_spec, act_spec, sess_mgr=sess_mgr, **kwargs)
        self.logger = StreamLogger(n_envs=n_envs, log_freq=10, sess_mgr=self.sess_mgr)

        self.start_step = self.start_step // self.n_epochs

    def minimize(self, advantages, returns):
        inputs = [a.reshape(-1, *a.shape[2:]) for a in self.obs + self.acts]
        tf_inputs = self.model.inputs + self.policy.inputs
        logli_old = self.sess_mgr.run(self.policy.logli, tf_inputs, inputs)

        inputs += [advantages.flatten(), returns.flatten(), logli_old, self.values.flatten()]
        tf_inputs += self.loss_inputs

        # TODO: rewrite this with persistent tensors to load data only once into the graph
        loss_terms = grads_norm = None
        n_samples = self.traj_len * self.batch_sz
        indices = np.arange(n_samples)
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for i in range(n_samples // self.minibatch_sz):
                idxs, idxe = i*self.minibatch_sz, (i+1)*self.minibatch_sz
                minibatch = [inpt[indices[idxs:idxe]] for inpt in inputs]
                loss_terms, grads_norm, *_ = self.sess_mgr.run(self.minimize_ops, tf_inputs, minibatch)

        return loss_terms, grads_norm

    def loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")
        logli_old = tf.placeholder(tf.float32, [None], name="logli_old")
        value_old = tf.placeholder(tf.float32, [None], name="value_old")

        ratio = tf.exp(self.policy.logli - logli_old)
        clipped_ratio = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)

        value_err = (self.value - returns)**2
        if self.clip_value > 0.0:
            clipped_value = tf.clip_by_value(self.value, value_old-self.clip_value, value_old+self.clip_value)
            clipped_value_err = (clipped_value - returns)**2
            value_err = tf.maximum(value_err, clipped_value_err)

        policy_loss = -tf.reduce_mean(tf.minimum(adv * ratio, adv * clipped_ratio))
        value_loss = tf.reduce_mean(value_err) * self.value_coef
        entropy_loss = tf.reduce_mean(self.policy.entropy) * self.entropy_coef
        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = policy_loss + value_loss - entropy_loss

        return full_loss, [policy_loss, value_loss, entropy_loss], [adv, returns, logli_old, value_old]
