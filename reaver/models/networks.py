import gin
import tensorflow as tf
import tensorflow.keras as K


@gin.configurable
def mlp(obs_spec, act_spec=None, with_value=True, layers=(64, 64), activation='relu'):
    assert act_spec or with_value

    inputs = [K.layers.Input(s.shape) for s in obs_spec]
    x = K.layers.Concatenate()(inputs) if len(inputs) > 1 else inputs[0]

    for size in layers:
        x = K.layers.Dense(size, activation=activation)(x)

    outputs = []

    if act_spec:
        outputs = [K.layers.Dense(s.size(), name="logits_" + s.name)(x) for s in act_spec]

    if with_value:
        value = K.layers.Dense(1, name="value_fn")(x)
        value = K.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(value)
        outputs.append(value)

    return K.Model(inputs=inputs, outputs=outputs)
