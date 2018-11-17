import tensorflow as tf
from tensorflow.keras.layers import Lambda


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
