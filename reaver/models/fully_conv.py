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

        self.logits = []
        for space in act_spec.spaces:
            if len(space.shape) == 1:
                # non-spatial action logits
                self.logits.append(L.Dense(space.shape[0], kernel_initializer='he_normal')(fc))
            else:
                self.logits.append(L.Conv2D(1, 1, **self.conv_cfg)(state))
                # flatten spatial logits, simplifying sampling
                self.logits[-1] = L.Flatten()(self.logits[-1])

        args_mask = tf.constant(act_spec.spaces[0].args_mask, dtype=tf.float32)

        # TODO replace with Autoregressive
        # TODO get rid of the index hard-code
        self.policy = MultiPolicy(self.logits, self.non_spatial_inputs[0], args_mask)


class MultiPolicy:
    def __init__(self, multi_logits, available_actions, args_mask):
        # tfp is really heavy on init, better to lazy load
        import tensorflow_probability as tfp
        # we can treat available_actions as a constant => no need to condition the distribution
        # large neg logit => normalized prob = 0 during sampling
        multi_logits[0] = tf.where(available_actions > 0, multi_logits[0], -1000 * tf.ones_like(multi_logits[0]))
        self.dists = [tfp.distributions.Categorical(logits) for logits in multi_logits]
        self.sample = [dist.sample() for dist in self.dists]

        self.entropy = sum([dist.entropy() for dist in self.dists])

        self.inputs = [tf.placeholder(tf.int32, [None]) for _ in self.dists]
        act_args_mask = tf.gather(args_mask, self.inputs[0])
        act_args_mask = tf.transpose(act_args_mask, [1, 0])

        self.logli = self.dists[0].log_prob(self.inputs[0])
        for i in range(1, len(self.dists)):
            self.logli += act_args_mask[i-1] * self.dists[i].log_prob(self.inputs[i])
        # tfp dists actually return neg log prob
        self.logli *= -1


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
            block[i] = tf.log(block[i] + 1e-10)

    block = tf.concat(block, axis=1)
    block = L.Conv2D(16, 5, activation='relu', **conv_cfg)(block)
    block = L.Conv2D(32, 3, activation='relu', **conv_cfg)(block)

    return block, inpt
