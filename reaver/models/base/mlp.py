import gin
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense
from reaver.models.base.layers import Squeeze, DenseWithVar


@gin.configurable
def build_mlp(obs_spec, act_spec, layer_sizes=(64, 64), activation='relu', value_separate=False):
    inputs = [Input(s.shape, name="input_" + s.name) for s in obs_spec]
    inputs_concat = Concatenate()(inputs) if len(inputs) > 1 else inputs[0]

    x = build_fc(inputs_concat, layer_sizes, activation)
    outputs = [logits_layer(space)(x) for space in act_spec]

    if value_separate:
        x = build_fc(inputs_concat, layer_sizes, activation, 'value_')

    value = Dense(1, name="value_out")(x)
    value = Squeeze(axis=-1)(value)
    outputs.append(value)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def logits_layer(space):
    if space.is_continuous():
        return DenseWithVar(space.size(), name="logits_" + space.name, var_name="logstd")
    return Dense(space.size(), name="logits_" + space.name)


def build_fc(input_layer, layer_sizes, activation, prefix=''):
    x = input_layer
    for i, size in enumerate(layer_sizes):
        x = Dense(size, activation=activation, name='%sfc%02d' % (prefix, i+1))(x)
    return x
