import gin
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense
from reaver.models.base.layers import Squeeze, Variable, RunningStatsNorm


@gin.configurable
def build_mlp(obs_spec, act_spec, layer_sizes=(64, 64), activation='relu', value_separate=False, obs_shift=False, obs_scale=False):
    inputs = inputs_ = [Input(s.shape, name="input_" + s.name) for s in obs_spec]
    if obs_shift or obs_scale:
        inputs_ = [RunningStatsNorm(obs_shift, obs_scale, name="norm_" + s.name)(x) for s, x in zip(obs_spec, inputs_)]
    inputs_concat = Concatenate()(inputs_) if len(inputs_) > 1 else inputs_[0]

    x = build_fc(inputs_concat, layer_sizes, activation)
    outputs = [build_logits(x, space) for space in act_spec]

    if value_separate:
        x = build_fc(inputs_concat, layer_sizes, activation, 'value_')

    value = Dense(1, name="value_out")(x)
    value = Squeeze(axis=-1)(value)
    outputs.append(value)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_logits(prev_layer, space):
    logits = Dense(space.size(), name="logits_" + space.name)(prev_layer)
    if space.is_continuous():
        logits = Variable(name="logstd")(logits)
    return logits


def build_fc(input_layer, layer_sizes, activation, prefix=''):
    x = input_layer
    for i, size in enumerate(layer_sizes):
        x = Dense(size, activation=activation, name='%sfc%02d' % (prefix, i+1))(x)
    return x
