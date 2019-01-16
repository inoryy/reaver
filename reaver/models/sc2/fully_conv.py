import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Concatenate, Dense, Embedding, Conv2D, Flatten, Lambda
from reaver.models.base.layers import Squeeze, Split, Transpose, Log, Broadcast2D


@gin.configurable
def build_fully_conv(obs_spec, act_spec, data_format='channels_first', broadcast_non_spatial=False, fc_dim=256):
    screen, screen_input = spatial_block('screen', obs_spec.spaces[0], conv_cfg(data_format, 'relu'))
    minimap, minimap_input = spatial_block('minimap', obs_spec.spaces[1], conv_cfg(data_format, 'relu'))

    non_spatial_inputs = [Input(s.shape) for s in obs_spec.spaces[2:]]

    if broadcast_non_spatial:
        non_spatial, spatial_dim = non_spatial_inputs[1], obs_spec.spaces[0].shape[1]
        non_spatial = Log()(non_spatial)
        broadcasted_non_spatial = Broadcast2D(spatial_dim)(non_spatial)
        state = Concatenate(axis=1, name="state_block")([screen, minimap, broadcasted_non_spatial])
    else:
        state = Concatenate(axis=1, name="state_block")([screen, minimap])

    fc = Flatten(name="state_flat")(state)
    fc = Dense(fc_dim, **dense_cfg('relu'))(fc)

    value = Dense(1, name="value_out", **dense_cfg(scale=0.1))(fc)
    value = Squeeze(axis=-1)(value)

    logits = []
    for space in act_spec:
        if space.is_spatial():
            logits.append(Conv2D(1, 1, **conv_cfg(data_format, scale=0.1))(state))
            logits[-1] = Flatten()(logits[-1])
        else:
            logits.append(Dense(space.size(), **dense_cfg(scale=0.1))(fc))

    mask_actions = Lambda(
        lambda x: tf.where(non_spatial_inputs[0] > 0, x, -1000 * tf.ones_like(x)),
        name="mask_unavailable_action_ids"
    )
    logits[0] = mask_actions(logits[0])

    return Model(
        inputs=[screen_input, minimap_input] + non_spatial_inputs,
        outputs=logits + [value]
    )


def spatial_block(name, space, cfg):
    inpt = Input(space.shape, name=name + '_input')
    block = Split(space.shape[0], axis=1)(inpt)

    for i, (name, dim) in enumerate(zip(space.spatial_feats, space.spatial_dims)):
        if dim > 1:
            block[i] = Squeeze(axis=1)(block[i])
            # Embedding dim 10 as per https://arxiv.org/pdf/1806.01830.pdf
            block[i] = Embedding(input_dim=dim, output_dim=10)(block[i])
            # [N, H, W, C] -> [N, C, H, W]
            block[i] = Transpose([0, 3, 1, 2])(block[i])
        else:
            block[i] = Log()(block[i])

    block = Concatenate(axis=1)(block)
    block = Conv2D(16, 5, **cfg)(block)
    block = Conv2D(32, 3, **cfg)(block)

    return block, inpt


def conv_cfg(data_format='channels_first', activation=None, scale=1.0):
    return dict(
        padding='same',
        activation=activation,
        data_format=data_format,
        kernel_initializer=VarianceScaling(scale=2.0*scale)
    )


def dense_cfg(activation=None, scale=1.0):
    return dict(
        activation=activation,
        kernel_initializer=VarianceScaling(scale=2.0*scale)
    )

