import tensorflow as tf
from tensorflow.keras.layers import Lambda, Layer


class RunningStatsNorm(Layer):
    """
    Normalizes inputs by running mean / std.dev statistics
    """
    def __init__(self, and_shift=True, and_scale=False, **kwargs):
        self.and_shift, self.and_scale = and_shift, and_scale
        self._ct = self._mu = self._var = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = (*input_shape[1:],)
        self._ct = self.add_weight('running_ct', (), initializer='zeros', trainable=False)
        self._mu = self.add_weight('running_mu', shape, initializer='zeros', trainable=False)
        self._var = self.add_weight('running_var', shape, initializer='ones', trainable=False)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        mu, var = self._update_stats(inputs)
        if self.and_shift:
            inputs -= mu
        if self.and_scale:
            inputs /= tf.sqrt(var)
        return inputs

    def _update_stats(self, x):
        ct = tf.maximum(1e-10, self._ct)

        ct_b = tf.to_float(tf.shape(x)[0])
        mu_b, var_b = tf.nn.moments(x, axes=[0])

        delta = mu_b - self._mu

        new_ct = ct + ct_b
        new_mu = self._mu + delta * ct_b / new_ct
        new_var = (self._var * ct + var_b * ct_b + delta ** 2 * ct * ct_b / new_ct) / new_ct

        self.add_update([
            tf.assign(self._ct, new_ct),
            tf.assign(self._mu, new_mu),
            tf.assign(self._var, new_var)
        ])

        return new_mu, new_var


class Variable(Layer):
    """
    Concatenate an extra trainable variable to the dense layer
    This variable is disconnected from the rest of the NN, including inputs
    """
    def __init__(self, **kwargs):
        self._var = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._var = self.add_weight('var', (1, input_shape[-1]), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        repeat _var N times to match inputs dim and then concatenate them
        """
        return tf.concat([inputs, tf.tile(self._var, (tf.shape(inputs)[0], 1))], axis=-1)


class Squeeze(Lambda):
    def __init__(self, axis=-1):
        Lambda.__init__(self, lambda x: tf.squeeze(x, axis=axis))


class Split(Lambda):
    def __init__(self, num_splits=2, axis=-1):
        Lambda.__init__(self, lambda x: tf.split(x, num_splits, axis=axis))


class Transpose(Lambda):
    def __init__(self, dims):
        Lambda.__init__(self, lambda x: tf.transpose(x, dims))


class Log(Lambda):
    def __init__(self):
        Lambda.__init__(self, lambda x: tf.log(x + 1e-10))


class Rescale(Lambda):
    def __init__(self, scale):
        Lambda.__init__(self, lambda x: tf.cast(x, tf.float32) * scale)


class Broadcast2D(Lambda):
    def __init__(self, size):
        Lambda.__init__(self, lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, 2), 3), [1, 1, size, size]))
