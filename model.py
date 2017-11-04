import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Dense, Flatten, Activation, Reshape


def cnn_block(cat_channels_in, cat_channels_out, num_channels):
    cat_inputs = Input(shape=(64, 64, cat_channels_in))
    num_inputs = Input(shape=(64, 64, num_channels))
    embed = Conv2D(cat_channels_out, 1)(cat_inputs)
    inputs = Concatenate(axis=3)([embed, num_inputs])
    conv1 = Conv2D(16, 5, padding='same', activation='relu')(inputs)
    conv2 = Conv2D(32, 3, padding='same', activation='relu')(conv1)
    return conv2, cat_inputs, num_inputs


def simple(screen_channels, minimap_channels):
    # todo non-spatial features, policy
    screen, s_cat_in, s_num_in = cnn_block(*screen_channels)
    minimap, m_cat_in, m_num_in = cnn_block(*minimap_channels)
    state = Concatenate(axis=3)([screen, minimap])
    spatial_action = Conv2D(1, 1, name='action_spatial')(state)
    spatial_action = Flatten()(spatial_action)
    spatial_action = Activation('softmax')(spatial_action)
    # spatial_action = Reshape((64, 64))(spatial_action)

    flat = Flatten()(state)
    fc1 = Dense(256, activation='relu')(flat)
    value = Dense(1, name='value')(fc1)

    return Model(inputs=[s_cat_in, s_num_in, m_cat_in, m_num_in], outputs=[spatial_action, value])
