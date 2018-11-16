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
        policy_coef=1.0,
        value_coef=0.5,
        entropy_coef=0.001,
    ):
        self.policy_coef = policy_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        SyncRunningAgent.__init__(self, n_envs)
        ActorCriticAgent.__init__(self, sess, obs_spec, act_spec, n_envs, traj_len)

    def _loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")

        policy_loss = -tf.reduce_mean(self.policy.logli * adv)
        value_loss = tf.reduce_mean((self.value - returns)**2)
        entropy_loss = tf.reduce_mean(self.policy.entropy)
        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = self.policy_coef*policy_loss + self.value_coef*value_loss - self.entropy_coef*entropy_loss

        return full_loss, [policy_loss, value_loss, entropy_loss, full_loss], [adv, returns]
