import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense


class DenseWithVar(Dense):
    """
    attaches an extra trainable variable to the dense layer as "extra_var"
    it is "invincible" to keras (e.g. missing in model summary)
    but in return keeps most of the glue code relatively simple

    example use case is mean + std for continuous control tasks where
    mean is defined as dense output from NN (inherited from Dense layer)
    std is defined as a separate (trainable!) variable
    """
    def __init__(self, units, var_name=None, **kwargs):
        self._var = None
        self._var_name = var_name
        super().__init__(units, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # have to do through backend to ensure it plays nice with Keras internally
        self._var = tf.keras.backend.variable([0.0]*self.units, name=self._name + "_" + self._var_name)

    def call(self, inputs):
        x = super().call(inputs)
        x.extra_var = self._var
        return x


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
