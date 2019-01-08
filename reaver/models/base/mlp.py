import gin
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense
from reaver.models.base.layers import Squeeze, Variable, RunningStatsNorm
from reaver.envs.base import Spec


@gin.configurable
def build_mlp(
        obs_spec: Spec,
        act_spec: Spec,
        layer_sizes=(64, 64),
        activation='relu',
        initializer='glorot_uniform',
        value_separate=False,
        obs_shift=False,
        obs_scale=False) -> tf.keras.Model:
    """
    Factory method for a simple fully connected neural network model used in e.g. MuJuCo environment

    If value separate is set to true then a separate path is added for value fn, otherwise branches out of last layer
    If obs shift is set to true then observations are normalized to mean zero with running mean estimate
    If obs scale is set to true then observations are standardized to std.dev one with running std.dev estimate
    """
    inputs = inputs_ = [Input(s.shape, name="input_" + s.name) for s in obs_spec]
    if obs_shift or obs_scale:
        inputs_ = [RunningStatsNorm(obs_shift, obs_scale, name="norm_" + s.name)(x) for s, x in zip(obs_spec, inputs_)]
    inputs_concat = Concatenate()(inputs_) if len(inputs_) > 1 else inputs_[0]

    x = build_fc(inputs_concat, layer_sizes, activation, initializer)
    outputs = [build_logits(space, x, initializer) for space in act_spec]

    if value_separate:
        x = build_fc(inputs_concat, layer_sizes, activation, initializer, 'value_')

    value = Dense(1, name="value_out", kernel_initializer=initializer)(x)
    value = Squeeze(axis=-1)(value)
    outputs.append(value)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_logits(space, prev_layer, initializer):
    logits = Dense(space.size(), kernel_initializer=initializer, name="logits_" + space.name)(prev_layer)
    if space.is_continuous():
        logits = Variable(name="logstd")(logits)
    return logits


def build_fc(input_layer, layer_sizes, activation, initializer, prefix=''):
    x = input_layer
    for i, size in enumerate(layer_sizes):
        x = Dense(size, activation=activation, kernel_initializer=initializer, name='%sfc%02d' % (prefix, i+1))(x)
    return x
