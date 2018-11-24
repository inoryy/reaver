import os
import math
import logging
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import zip_longest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot_from_summaries(summaries_path, title=None, samples_per_update=512, updates_per_log=100):
    acc = EventAccumulator(summaries_path)
    acc.Reload()

    rews_mean = np.array([s[2] for s in acc.Scalars('Rewards/Mean')])
    rews_std = np.array([s[2] for s in acc.Scalars('Rewards/Std')])
    x = samples_per_update * updates_per_log * np.arange(0, len(rews_mean))

    if not title:
        title = summaries_path.split('/')[-1].split('_')[0]

    plt.plot(x, rews_mean)
    plt.fill_between(x, rews_mean - rews_std, rews_mean + rews_std, alpha=0.2)
    plt.xlabel('Samples')
    plt.ylabel('Episode Rewards')
    plt.title(title)
    plt.xlim([0, x[-1]+1])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    parser.add_argument('--titles', nargs='*', default=[])
    parser.add_argument('--samples_per_update', type=int, default=512)
    parser.add_argument('--updates_per_log', type=int, default=100)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    plt.style.use('seaborn')
    mpl.rcParams['figure.figsize'] = (10, 10)

    paths = [os.path.join(args.logdir, p) for p in os.listdir(args.logdir)]

    assert len(paths) >= len(args.titles), "More titles than summaries"

    n_plots = len(paths)
    n_rows = math.ceil(n_plots / 2)
    for idx, (path, title) in enumerate(zip_longest(paths, args.titles)):
        if n_plots > 1:
            plt.subplot(n_rows, 2, 1 + idx)
        plot_from_summaries(path, title, args.samples_per_update, args.updates_per_log)
    plt.tight_layout()
    plt.show()
