import tensorflow as tf
import tensorflow.keras.layers as L


class FullyConv:
    def __init__(self, obs_spec, act_spec):
        # TODO: NCHW is only supported on GPU atm, make this device agnostic
        self.conv_cfg = dict(padding='same', data_format='channels_first')
        screen, self.screen_input = spatial_block(obs_spec.spaces[0], self.conv_cfg)
        minimap, self.minimap_input = spatial_block(obs_spec.spaces[1], self.conv_cfg)

        state = tf.concat([screen, minimap], axis=1)
        fc = L.Flatten()(state)
        fc = L.Dense(256, activation='relu')(fc)

        self.value = L.Dense(1)(fc)

        # note: does not produce probability distributions, only logits
        self.policy = []
        for space in act_spec.spaces:
            if len(space.shape) == 1:
                self.policy.append(L.Dense(space.shape[0])(fc))
            else:
                # spatial policy, but flatten it to 1d for simplicity
                self.policy.append(L.Flatten()(L.Conv2D(1, 1, **self.conv_cfg)(state)))

        self.inputs = [self.screen_input, self.minimap_input]
        # TODO connect non-spatial inputs to the network
        self.inputs += [L.Input(s.shape) for s in obs_spec.spaces[2:]]


def spatial_block(space, conv_cfg):
    inpt = L.Input(space.shape)

    block = tf.split(inpt, space.shape[0], axis=1)
    for i, (name, dim) in enumerate(zip(space.spatial_feats, space.spatial_dims)):
        if dim > 1:
            # categorical spatial feature
            block[i] = tf.squeeze(block[i], axis=1)
            block[i] = L.Embedding(input_dim=dim, output_dim=2)(block[i])
            # [N, H, W, C] -> [N, C, H, W]
            block[i] = tf.transpose(block[i], [0, 3, 1, 2])
        else:
            # TODO: do I need + 1.0 here?
            block[i] = tf.log(block[i] + 1.0)

    block = tf.concat(block, axis=1)
    block = L.Conv2D(16, 5, activation='relu', **conv_cfg)(block)
    block = L.Conv2D(32, 3, activation='relu', **conv_cfg)(block)

    return block, inpt
