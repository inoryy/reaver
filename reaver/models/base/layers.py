import tensorflow as tf
from tensorflow.keras.layers import Lambda, Layer


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
        return tf.concat([inputs, tf.tile(self._var, tf.shape(inputs))], axis=-1)


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
