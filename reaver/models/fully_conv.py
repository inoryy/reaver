import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_probability as tfp


class FullyConv:
    def __init__(self, obs_spec, act_spec):
        # TODO NCHW is only supported on GPU atm, make this device agnostic
        self.conv_cfg = dict(padding='same', data_format='channels_first')
        screen, self.screen_input = spatial_block(obs_spec.spaces[0], self.conv_cfg)
        minimap, self.minimap_input = spatial_block(obs_spec.spaces[1], self.conv_cfg)

        # TODO connect non-spatial inputs to the state block
        self.non_spatial_inputs = [L.Input(s.shape) for s in obs_spec.spaces[2:]]
        self.inputs = [self.screen_input, self.minimap_input] + self.non_spatial_inputs

        state = tf.concat([screen, minimap], axis=1)
        fc = L.Flatten()(state)
        fc = L.Dense(256, activation='relu')(fc)

        self.value = L.Dense(1)(fc)

        self.logits = []
        for space in act_spec.spaces:
            if len(space.shape) == 1:
                self.logits.append(L.Dense(space.shape[0])(fc))
            else:
                self.logits.append(L.Flatten()(L.Conv2D(1, 1, **self.conv_cfg)(state)))

        # TODO replace with Autoregressive
        # TODO get rid of the index hard-code
        self.policy = MultiPolicy(self.logits, self.non_spatial_inputs[0])


class MultiPolicy:
    def __init__(self, multi_logits, available_actions):
        # we can treat available_actions as a constant => no need to condition the distribution
        # TODO check if this actually masks properly
        # large neg number => normalized prob --> 0
        multi_logits[0] = tf.where(available_actions > 0, multi_logits[0], -10000 * tf.ones_like(multi_logits[0]))
        self.dists = [tfp.distributions.Categorical(logits) for logits in multi_logits]

        self.sample = [dist.sample() for dist in self.dists]
        # TODO push individual entropy / logli to summary
        self.entropy = sum([dist.entropy() for dist in self.dists])

        self.action_input = [L.Input(shape=(1,), dtype=tf.int32) for _ in self.dists]
        # TODO get rid of this tf.squeeze() due to shape (1,)
        action = [tf.squeeze(act) for act in self.action_input]
        self.logli = -sum([dist.log_prob(act) for dist, act in zip(self.dists, action)])



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
            # TODO do I need + 1.0 here?
            block[i] = tf.log(block[i] + 1.0)

    block = tf.concat(block, axis=1)
    block = L.Conv2D(16, 5, activation='relu', **conv_cfg)(block)
    block = L.Conv2D(32, 3, activation='relu', **conv_cfg)(block)

    return block, inpt
