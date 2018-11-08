import tensorflow as tf
import tensorflow.keras.layers as L


class BasicNN:
    def __init__(self, obs_spec, act_spec, n_layers=2, size=64, activation='tanh'):
        # basic nn shouldn't deal with more complicated cases
        assert len(obs_spec.spaces) == 1
        for sp in act_spec.spaces:
            if len(sp.shape) == 0 and sp.categorical:
                sp.shape = (sp.hi,)
            assert len(sp.shape) == 1

        self.inputs = [L.Input(s.shape) for s in obs_spec.spaces]

        cfg = dict(kernel_initializer='he_normal')

        x = self.inputs[0]
        for _ in range(n_layers):
            x = L.Dense(size, activation=activation, **cfg)(x)

        self.logits = [L.Dense(s.shape[0], **cfg)(x) for s in act_spec.spaces]
        self.policy = MultiPolicy(self.logits)
        self.value = tf.squeeze(L.Dense(1, **cfg)(x), axis=-1)


class MultiPolicy:
    def __init__(self, multi_logits):
        # tfp is really heavy on init, better to lazy load
        import tensorflow_probability as tfp

        self.dists = [tfp.distributions.Categorical(logits) for logits in multi_logits]
        self.sample = [dist.sample() for dist in self.dists]

        self.action_inputs = [tf.placeholder(tf.int32, [None]) for _ in self.dists]
        # TODO push individual entropy / logli to summary
        self.entropy = sum([dist.entropy() for dist in self.dists])
        self.logli = -sum([dist.log_prob(act) for dist, act in zip(self.dists, self.action_inputs)])
