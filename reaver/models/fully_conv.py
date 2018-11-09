import tensorflow as tf
import tensorflow.keras.layers as L


# TODO extend from Keras.models?
class FullyConv:
    def __init__(self, obs_spec, act_spec):
        # TODO add support for LSTM via TimeDistributed
        # TODO NCHW is only supported on GPU atm, make this device agnostic
        self.conv_cfg = dict(padding='same', data_format='channels_first', kernel_initializer='he_normal')
        screen, self.screen_input = spatial_block(obs_spec.spaces[0], self.conv_cfg)
        minimap, self.minimap_input = spatial_block(obs_spec.spaces[1], self.conv_cfg)

        # TODO connect non-spatial inputs to the state block
        self.non_spatial_inputs = [L.Input(s.shape) for s in obs_spec.spaces[2:]]
        self.inputs = [self.screen_input, self.minimap_input] + self.non_spatial_inputs

        state = tf.concat([screen, minimap], axis=1)
        fc = L.Flatten()(state)
        fc = L.Dense(256, activation='relu', kernel_initializer='he_normal')(fc)

        # TODO do I really want to squeeze here?
        self.value = tf.squeeze(L.Dense(1, kernel_initializer='he_normal')(fc), axis=-1)

        # TODO only flow gradients to arg logits that actually contributed to the action
        self.logits = []
        for space in act_spec.spaces:
            if len(space.shape) == 1:
                # non-spatial action logits
                self.logits.append(L.Dense(space.shape[0], kernel_initializer='he_normal')(fc))
            else:
                self.logits.append(L.Conv2D(1, 1, **self.conv_cfg)(state))
                # flatten spatial logits, simplifying sampling
                self.logits[-1] = L.Flatten()(self.logits[-1])

        # TODO replace with Autoregressive
        # TODO get rid of the index hard-code
        self.policy = MultiPolicy(self.logits, self.non_spatial_inputs[0])


class MultiPolicy:
    def __init__(self, multi_logits, available_actions):
        # tfp is really heavy on init, better to lazy load
        import tensorflow_probability as tfp
        # we can treat available_actions as a constant => no need to condition the distribution
        # large neg logit => normalized prob = 0 during sampling
        multi_logits[0] = tf.where(available_actions > 0, multi_logits[0], -1000 * tf.ones_like(multi_logits[0]))
        self.dists = [tfp.distributions.Categorical(logits) for logits in multi_logits]
        self.sample = [dist.sample() for dist in self.dists]

        # TODO push individual entropy / logli to summary
        # TODO mask entropy of unused args
        self.entropy = sum([dist.entropy() for dist in self.dists])

        self.action_inputs = [tf.placeholder(tf.int32, [None]) for _ in self.dists]
        self.logli = -sum([dist.log_prob(act) for dist, act in zip(self.dists, self.action_inputs)])


def spatial_block(space, conv_cfg):
    inpt = L.Input(space.shape)

    block = tf.split(inpt, space.shape[0], axis=1)
    for i, (name, dim) in enumerate(zip(space.spatial_feats, space.spatial_dims)):
        if dim > 1:
            # categorical spatial feature
            block[i] = tf.squeeze(block[i], axis=1)
            block[i] = L.Embedding(input_dim=dim, output_dim=1, embeddings_initializer='he_normal')(block[i])
            # [N, H, W, C] -> [N, C, H, W]
            block[i] = tf.transpose(block[i], [0, 3, 1, 2])
        else:
            # TODO do I need + 1.0 here?
            block[i] = tf.log(block[i] + 1.0)

    block = tf.concat(block, axis=1)
    block = L.Conv2D(16, 5, activation='relu', **conv_cfg)(block)
    block = L.Conv2D(32, 3, activation='relu', **conv_cfg)(block)

    return block, inpt
