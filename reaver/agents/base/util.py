import sys
import time
import gin
import numpy as np
import tensorflow as tf
from collections import deque


def discounted_cumsum(x, discount):
    y = np.zeros_like(x)
    y[-1] = x[-1]
    for t in range(x.shape[0]-2, -1, -1):
        y[t] = x[t] + discount[t] * y[t+1]
    return y


def tf_run(sess, tf_op, tf_inputs, inputs):
    return sess.run(tf_op, feed_dict=dict(zip(tf_inputs, inputs)))


@gin.configurable
class AgentLogger:
    def __init__(self, agent, n_updates=100, n_detailed=10, verbosity=3, summary_logs_dir='data/summaries'):
        self.agent, self.verbosity = agent, verbosity
        self.n_updates, self.n_detailed = n_updates, n_detailed
        self.env_eps = [0]*self.agent.n_envs
        self.env_rews = [0]*self.agent.n_envs
        self.n_eps = max(10, self.agent.n_envs)
        self.tot_rews = deque([], maxlen=self.n_eps)
        self.writer = create_summary_writer(summary_logs_dir)

    def on_step(self, step):
        t = step % self.agent.traj_len
        self.env_rews += self.agent.rewards[t]
        for i in range(self.agent.n_envs):
            if self.agent.dones[t, i]:
                self.tot_rews.append(self.env_rews[i])
                self.env_rews[i] = 0.
                self.env_eps[i] += 1

    def on_update(self, step, loss_terms, grads_norm, returns, adv, next_value):
        if self.verbosity < 1:
            return

        update_step = (step+1) // self.agent.traj_len
        if update_step > 1 and update_step % self.n_updates:
            return

        loss_terms = np.array(loss_terms).round(5)
        np.set_printoptions(suppress=True, precision=3)

        print("######################################################")
        runtime = max(1, int(time.time() - self.agent.start_time))
        frames = (step+1) * self.agent.n_envs

        print("Runner Stats:")
        print("Time    ", runtime)
        print("Eps     ", int(np.sum(self.env_eps)))
        print("Frames  ", frames)
        print("Steps   ", step+1)
        print("Updates ", update_step)
        print("FPS     ", frames // runtime)

        tot_rews = self.tot_rews if len(self.tot_rews) > 0 else [0]
        rews = [np.mean(tot_rews), np.std(tot_rews), np.min(tot_rews), np.max(tot_rews)]
        print()
        print("Total Rewards For Last %d Eps:" % self.n_eps)
        print("Mean %.3f" % rews[0])
        print("Std  %.3f" % rews[1])
        print("Min  %.3f" % rews[2])
        print("Max  %.3f" % rews[3])
        add_summaries(self.writer, ['Mean', 'Std', 'Min', 'Max'], rews, update_step, 'Rewards')

        if self.verbosity < 2:
            return

        print()
        print("Losses For Last Update:")
        print("Policy loss  ", loss_terms[0])
        print("Value loss   ", loss_terms[1])
        print("Entropy loss ", loss_terms[2])
        print("Grads norm   ", grads_norm)
        add_summaries(self.writer, ['Policy', 'Value', 'Entropy'], loss_terms, update_step, 'Losses')
        add_summary(self.writer, 'Grads', grads_norm, update_step, 'Losses')

        if self.verbosity < 3:
            return

        np.set_printoptions(suppress=True, precision=2)
        n_steps = min(self.n_detailed, self.agent.traj_len)

        print()
        print("First Env For Last %d Steps:" % n_steps)
        print("Dones      ", self.agent.dones[-n_steps:, 0].flatten().astype(int))
        print("Rewards    ", self.agent.rewards[-n_steps:, 0].flatten())
        print("Values     ", self.agent.values[-n_steps:, 0].flatten(), round(next_value[0], 3))
        print("Returns    ", returns[-n_steps:, 0].flatten())
        print("Advs       ", adv[-n_steps:, 0].flatten())

        if self.verbosity >= 4:
            logits = tf_run(self.agent.sess, self.agent.policy.logits[0], self.agent.model.inputs,
                                       [o[-n_steps:, 0] for o in self.agent.obs])
            action_ids = self.agent.acts[0][-n_steps:, 0].flatten()

            print("Action ids ", action_ids)
            print("Act logits ", logits[np.arange(n_steps), action_ids])

        if self.verbosity >= 5:
            print()
            print("Note: action ids listed are not equivalent to pysc2")
            for t in range(n_steps-1, -1, -1):
                trv = n_steps - t
                avail = np.argwhere(self.agent.obs[2][-trv, 0]).flatten()
                avail_logits = logits[t, avail]
                avail_sorted = np.argsort(avail_logits)
                print("Step", -trv+1)
                print("Actions   ", self.agent.obs[2][-trv, 0].sum())
                print("Action ids", avail[avail_sorted[:3]], "..."*3, avail[avail_sorted[-5:]])
                print("Logits    ", avail_logits[avail_sorted[:3]], "...", avail_logits[avail_sorted[-5:]])
            print("######################################################")

        sys.stdout.flush()


# https://github.com/tambetm/TSCL/blob/master/addition/tensorboard_utils.py
def create_summary_writer(logdir):
    return tf.summary.FileWriter(logdir)


def create_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def add_summary(writer, tag, value, step, prefix=''):
    writer.add_summary(create_summary(prefix + '/' + tag, value), global_step=step)


def add_summaries(writer, tags, values, step, prefix=''):
    for (t, v) in zip(tags, values):
        s = create_summary(prefix + '/' + t, v)
        writer.add_summary(s, global_step=step)
