import gin.tf
import tensorflow as tf

from . import SyncRunningAgent, ActorCriticAgent


@gin.configurable
class AdvantageActorCriticAgent(SyncRunningAgent, ActorCriticAgent):
    def __init__(
        self,
        sess,
        obs_spec,
        act_spec,
        n_envs=4,
        traj_len=16,
        value_coef=0.5,
        entropy_coef=0.001,
    ):
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        SyncRunningAgent.__init__(self, n_envs)
        ActorCriticAgent.__init__(self, sess, obs_spec, act_spec, n_envs, traj_len)

    def _loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")

        policy_loss = -tf.reduce_mean(self.policy.logli * adv)
        value_loss = tf.reduce_mean((self.value - returns)**2) * self.value_coef
        entropy_loss = tf.reduce_mean(self.policy.entropy) * self.entropy_coef
        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = policy_loss + value_loss - entropy_loss

        return full_loss, [policy_loss, value_loss, entropy_loss], [adv, returns]
