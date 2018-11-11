import tensorflow as tf
from . import SyncRunningAgent, ActorCriticAgent


class AdvantageActorCriticAgent(SyncRunningAgent, ActorCriticAgent):
    def __init__(self, model_cls, obs_spec, act_spec, n_envs=4, batch_sz=16, **_kwargs):
        SyncRunningAgent.__init__(self, n_envs)

        kwargs = dict(
            policy_coef=1.0,
            value_coef=0.5,
            entropy_coef=0.001,
        )
        if _kwargs:
            kwargs.update(_kwargs)

        ActorCriticAgent.__init__(self, model_cls, obs_spec, act_spec, (batch_sz, n_envs), **kwargs)

    def _loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")

        policy_loss = -tf.reduce_mean(self.model.policy.logli * adv)
        value_loss = tf.reduce_mean((self.model.value - returns)**2)
        entropy_loss = tf.reduce_mean(self.model.policy.entropy)
        loss_terms = [policy_loss, value_loss, entropy_loss]

        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = self.kwargs['policy_coef']*policy_loss \
            + self.kwargs['value_coef']*value_loss \
            - self.kwargs['entropy_coef']*entropy_loss

        return full_loss, loss_terms + [full_loss], [adv, returns]
