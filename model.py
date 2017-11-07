import tensorflow as tf
from tensorflow.contrib import layers


def cnn_block(cat_channels_in, cat_channels_out, num_channels):
    cat_inputs = tf.placeholder(tf.float32, [None, 64, 64, cat_channels_in])
    num_inputs = tf.placeholder(tf.float32, [None, 64, 64, num_channels])

    embed = layers.conv2d(inputs=cat_inputs, num_outputs=cat_channels_out, kernel_size=1)
    inputs = tf.concat([embed, num_inputs], axis=3)

    conv1 = layers.conv2d(inputs=inputs, num_outputs=16, kernel_size=5)
    conv2 = layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=3)

    return conv2, cat_inputs, num_inputs


def fully_conv(screen_channels, minimap_channels):
    # todo non-spatial features, policy
    screen, s_cat_in, s_num_in = cnn_block(*screen_channels)
    minimap, m_cat_in, m_num_in = cnn_block(*minimap_channels)
    state = tf.concat([screen, minimap], axis=3)
    spatial_policy = layers.conv2d(inputs=state, num_outputs=1, kernel_size=1, activation_fn=None)
    spatial_policy = tf.nn.softmax(layers.flatten(spatial_policy))

    flat_state = layers.flatten(state)
    fc1 = layers.fully_connected(flat_state, num_outputs=256)
    value = tf.squeeze(layers.fully_connected(fc1, num_outputs=1, activation_fn=None), axis=1)

    return [s_cat_in, s_num_in, m_cat_in, m_num_in], [spatial_policy, value]
