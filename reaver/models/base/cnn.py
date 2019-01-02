import gin
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Conv2D, Flatten
from reaver.models.base.layers import Squeeze, Rescale, Transpose, RunningStatsNorm


@gin.configurable
def build_cnn_nature(obs_spec, act_spec, data_format='channels_first', value_separate=False, obs_shift=False, obs_scale=False):
    conv_cfg = dict(padding='same', data_format=data_format, activation='relu')
    conv_spec = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]

    inputs = [Input(s.shape, name="input_" + s.name) for s in obs_spec]
    inputs_concat = Concatenate()(inputs) if len(inputs) > 1 else inputs[0]

    # expected NxCxHxW, but got NxHxWxC
    if data_format == 'channels_first' and inputs_concat.shape[1] > 3:
        inputs_concat = Transpose([0, 3, 1, 2])(inputs_concat)

    inputs_scaled = Rescale(1./255)(inputs_concat)
    if obs_shift or obs_scale:
        inputs_scaled = RunningStatsNorm(obs_shift, obs_scale)(inputs_scaled)

    x = build_cnn(inputs_scaled, conv_spec, conv_cfg, dense=512, prefix='policy_')
    outputs = [Dense(s.size(), name="logits_" + s.name)(x) for s in act_spec]

    if value_separate:
        x = build_cnn(inputs_scaled, conv_spec, conv_cfg, dense=512, prefix='value_')

    value = Dense(1, name="value_out")(x)
    value = Squeeze(axis=-1)(value)
    outputs.append(value)

    return Model(inputs=inputs, outputs=outputs)


def build_cnn(input_layer, layers, conv_cfg, dense=None, prefix=''):
    x = input_layer
    for i, (n_filters, kernel_size, stride) in enumerate(layers):
        x = Conv2D(n_filters, kernel_size, stride, name='%sconv%02d' % (prefix, i+1), **conv_cfg)(x)

    if dense:
        x = Flatten()(x)
        x = Dense(dense)(x)

    return x
