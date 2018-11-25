import gin.tf
import tensorflow as tf

from reaver.utils import Logger
from reaver.models import build_mlp, MultiPolicy
from reaver.agents.base import SyncRunningAgent, ActorCriticAgent


@gin.configurable
class AdvantageActorCriticAgent(SyncRunningAgent, ActorCriticAgent):
    def __init__(
        self,
        obs_spec,
        act_spec,
        model_fn=build_mlp,
        policy_cls=MultiPolicy,
        sess_mgr=None,
        n_envs=4,
        traj_len=16,
        batch_sz=16,
        discount=0.99,
        gae_lambda=0.95,
        clip_rewards=0.0,
        normalize_advantages=True,
        bootstrap_terminals=False,
        clip_grads_norm=0.0,
        value_coef=0.5,
        entropy_coef=0.001,
        optimizer=tf.train.AdamOptimizer(),
        logger=Logger(),
    ):
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        SyncRunningAgent.__init__(self, n_envs)
        ActorCriticAgent.__init__(
            self, obs_spec, act_spec, model_fn, policy_cls, sess_mgr, traj_len, batch_sz, discount,
            gae_lambda, clip_rewards, normalize_advantages, bootstrap_terminals, clip_grads_norm, optimizer, logger
        )

    def loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")

        policy_loss = -tf.reduce_mean(self.policy.logli * adv)
        value_loss = tf.reduce_mean((self.value - returns)**2) * self.value_coef
        entropy_loss = tf.reduce_mean(self.policy.entropy) * self.entropy_coef
        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = policy_loss + value_loss - entropy_loss

        return full_loss, [policy_loss, value_loss, entropy_loss], [adv, returns]
