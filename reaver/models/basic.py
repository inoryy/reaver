import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L


class BasicNN:
    def __init__(self, obs_spec, act_spec, n_layers=2, size=64, activation='tanh'):
        # basic nn shouldn't deal with more complicated cases
        assert len(obs_spec.spaces) == 1

        cfg = dict(kernel_initializer='he_normal')

        self.inputs = [L.Input(s.shape) for s in obs_spec.spaces]
        x = self.inputs[0]
        for _ in range(n_layers):
            x = L.Dense(size, activation=activation, **cfg)(x)

        self.logits = [L.Dense(s.size(), **cfg)(x) for s in act_spec.spaces]
        self.policy = MultiPolicy(act_spec, self.logits)
        self.value = tf.squeeze(L.Dense(1, **cfg)(x), axis=-1)


class MultiPolicy:
    def __init__(self, act_spec, multi_logits):
        # tfp is really heavy on init, better to lazy load
        import tensorflow_probability as tfp

        def make_dist(space, logits):
            if space.is_continuous():
                return tfp.distributions.MultivariateNormalDiag(logits)
            else:
                return tfp.distributions.Categorical(logits)

        self.inputs = [tf.placeholder(s.dtype, [None, *s.shape]) for s in act_spec.spaces]
        self.dists = [make_dist(s, l) for s, l in zip(act_spec.spaces, multi_logits)]

        self.entropy = sum([dist.entropy() for dist in self.dists])
        self.logli = -sum([dist.log_prob(act) for dist, act in zip(self.dists, self.inputs)])

        self.sample = [dist.sample() for dist in self.dists]
