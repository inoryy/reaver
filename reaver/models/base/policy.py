import gin
import tensorflow as tf


@gin.configurable
class MultiPolicy:
    def __init__(self, act_spec, logits):
        self.logits = logits
        self.inputs = [tf.placeholder(s.dtype, [None, *s.shape]) for s in act_spec]

        self.dists = [self.make_dist(s, l) for s, l in zip(act_spec.spaces, logits)]

        self.entropy = sum([dist.entropy() for dist in self.dists])
        self.logli = sum([dist.log_prob(act) for dist, act in zip(self.dists, self.inputs)])

        self.sample = [dist.sample() for dist in self.dists]

    @staticmethod
    def make_dist(space, logits):
        # tfp is really heavy on init, better to lazy load
        import tensorflow_probability as tfp

        if space.is_continuous():
            mu, logstd = tf.split(logits, 2, axis=-1)
            return tfp.distributions.MultivariateNormalDiag(mu, tf.exp(logstd))
        else:
            return tfp.distributions.Categorical(logits)
