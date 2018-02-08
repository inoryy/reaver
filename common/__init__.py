import numpy as np
import tensorflow as tf

from pysc2.lib import features


def n_channels(feats):
    return n_channels_type('SCREEN', feats['screen']), n_channels_type('MINIMAP', feats['minimap'])


def n_channels_type(type, feat_names):
    cat_channels_in = cat_channels_out = num_channels = 0
    feats = getattr(features, type + '_FEATURES')
    for f_name in feat_names:
        f = getattr(feats, f_name)
        if f.type == features.FeatureType.CATEGORICAL:
            cat_channels_out += 1
            cat_channels_in += f.scale
        else:
            num_channels += 1

        # ignore background noise
        # TODO do this abstractly
        if f.name == 'player_relative':
            cat_channels_in -= 1
    return cat_channels_in, cat_channels_out, num_channels


def preprocess_inputs(x, feats):
    return preprocess_type(x, 'screen', feats) + preprocess_type(x, 'minimap', feats)


def preprocess_type(x, type, feats):
    cat_x, num_x = zip(*[preprocess_obs(obs, type, feats[type]) for obs in x])
    return np.array(cat_x), np.array(num_x)


def preprocess_obs(obs, type, feat_names):
    cat, num = [], []
    feats = getattr(features, type.upper() + '_FEATURES')
    for f_name in feat_names:
        f = getattr(feats, f_name)
        if f.type == features.FeatureType.CATEGORICAL:
            cat.append(one_hot(obs.observation[type][f.index], f.scale))
            # ignore background noise
            # TODO do this abstractly
            if f.name == 'player_relative':
                cat[-1] = cat[-1][:, :, 1:]
        else:
            num.append(obs.observation[type][f.index])

    cat = np.concatenate(cat, axis=2) if len(cat) > 0 else []
    num = np.array(num).transpose((1, 2, 0)) if len(num) > 0 else []

    return cat, num


def one_hot(x, n_classes):
    out = np.zeros((x.size, n_classes), dtype=np.uint8)
    out[np.arange(x.size), x.ravel()] = 1
    out.shape = x.shape + (n_classes,)
    return out


def unravel_coords(action, sz=(64, 64)):
    return list(zip(*np.unravel_index(action, dims=sz)))
