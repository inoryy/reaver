import tensorflow as tf
from tensorflow.contrib import layers


def fully_conv(config):
    screen, screen_input = cnn_block(config.sz, config.screen_dims(), config.embed_dim_fn)
    minimap, minimap_input = cnn_block(config.sz, config.minimap_dims(), config.embed_dim_fn)
    non_spatial, non_spatial_inputs = non_spatial_block(config.sz, config.non_spatial_dims(), config.ns_idx)

    state = tf.concat([screen, minimap, non_spatial], axis=1)
    fc1 = layers.fully_connected(layers.flatten(state), num_outputs=256)
    value = tf.squeeze(layers.fully_connected(fc1, num_outputs=1, activation_fn=None), axis=1)

    # TODO mask unused args
    policy = []
    for dim, is_spatial in config.policy_dims():
        if is_spatial:
            logits = layers.conv2d(state, num_outputs=1, kernel_size=1, activation_fn=None, data_format="NCHW")
            policy.append(tf.nn.softmax(layers.flatten(logits)))
        else:
            policy.append(layers.fully_connected(fc1, num_outputs=dim, activation_fn=tf.nn.softmax))
    policy[0] = mask_probs(policy[0], non_spatial_inputs[config.ns_idx['available_actions']])

    return [policy, value], [screen_input, minimap_input] + non_spatial_inputs


def cnn_block(sz, dims, embed_dim_fn):
    block_input = tf.placeholder(tf.float32, [None, sz, sz, len(dims)])
    block = tf.transpose(block_input, [0, 3, 1, 2]) # NHWC -> NCHW

    block = tf.split(block, len(dims), axis=1)
    for i, d in enumerate(dims):
        if d > 1:
            block[i] = tf.one_hot(tf.to_int32(tf.squeeze(block[i], axis=1)), d, axis=1)
            block[i] = layers.conv2d(block[i], num_outputs=embed_dim_fn(d), kernel_size=1, data_format="NCHW")
        else:
            block[i] = tf.log(block[i] + 1.0)
    block = tf.concat(block, axis=1)

    conv1 = layers.conv2d(block, num_outputs=16, kernel_size=5, data_format="NCHW")
    conv2 = layers.conv2d(conv1, num_outputs=32, kernel_size=3, data_format="NCHW")

    return conv2, block_input


def non_spatial_block(sz, dims, idx):
    block_inputs = [tf.placeholder(tf.float32, [None, *dim]) for dim in dims]
    # TODO currently too slow with full inputs
    # block = [broadcast(block_inputs[i], sz) for i in range(len(dims))]
    # block = tf.concat(block, axis=1)
    block = broadcast(tf.log(block_inputs[idx['player']] + 1.0), sz)
    return block, block_inputs


# based on https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/networks/fully_conv.py#L91-L96
def broadcast(tensor, sz):
    return tf.tile(tf.expand_dims(tf.expand_dims(tensor, 2), 3), [1, 1, sz, sz])


def mask_probs(probs, mask):
    masked = probs * mask
    return masked / tf.reduce_sum(masked, axis=1, keep_dims=True)
