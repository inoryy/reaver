import os
import sys
import time
import numpy as np
from collections import deque, namedtuple


class Logger:
    def on_start(self): ...

    def on_step(self, step, rewards, dones): ...

    def on_update(self, step, loss_terms, grads_norm, returns, adv, next_value): ...

    def on_finish(self): ...


class StreamLogger(Logger):
    def __init__(self, n_envs, log_freq=100, rew_avg_eps=100, sess_mgr=None, log_file_path=None):
        self.n_envs = n_envs
        self.log_freq = log_freq
        self.rew_avg_eps = rew_avg_eps

        self.env_eps = [0]*n_envs
        self.env_rews = [0]*n_envs
        self.ep_rews_sum = deque([], maxlen=self.rew_avg_eps)

        self.run_time = 0
        self.start_time = None

        self.sess_mgr = sess_mgr
        self.streams = [sys.stdout]
        self.log_file_path = None
        if self.sess_mgr.training_enabled:
            self.log_file_path = log_file_path

        ColumnParams = namedtuple("ColumnParams", ["abbr", "width", "precision"])
        self.col_params = dict(
            runtime=ColumnParams("T", 6, 0),
            frames=ColumnParams("Fr", 9, 0),
            episodes=ColumnParams("Ep", 6, 0),
            updates=ColumnParams("Up", 6, 0),
            ep_rews_mean=ColumnParams("RMe", 7, 2),
            ep_rews_std=ColumnParams("RSd", 7, 2),
            ep_rews_max=ColumnParams("RMa", 7, 2),
            ep_rews_min=ColumnParams("RMi", 7, 2),
            policy_loss=ColumnParams("Pl", 8, 3),
            value_loss=ColumnParams("Vl", 8, 3),
            entropy_loss=ColumnParams("El", 6, 4),
            grads_norm=ColumnParams("Gr", 8, 3),
            frames_per_second=ColumnParams("Fps", 5, 0),
        )

        self.col_fmt = "| {abbr} {value:{width}.{precision}f} "

    def on_step(self, step, rewards, dones):
        self.env_rews += rewards
        for i in range(self.n_envs):
            if not dones[i]:
                continue
            self.ep_rews_sum.append(self.env_rews[i])
            self.env_rews[i] = 0
            self.env_eps[i] += 1

    def on_update(self, step, loss_terms, grads_norm, returns, adv, next_value):
        if step > 1 and step % self.log_freq:
            return

        frames = step * np.prod(returns.shape)
        run_time = max(1, int(time.time() - self.start_time)) + self.run_time
        ep_rews = np.array(self.ep_rews_sum or [0])

        logs = dict(
            runtime=run_time,
            frames=frames,
            updates=step,
            episodes=int(np.sum(self.env_eps)),
            frames_per_second=frames // run_time,
            ep_rews_mean=ep_rews.mean(),
            ep_rews_std=ep_rews.std(),
            ep_rews_max=ep_rews.max(),
            ep_rews_min=ep_rews.min(),
            policy_loss=loss_terms[0],
            value_loss=loss_terms[1],
            entropy_loss=loss_terms[2],
            grads_norm=grads_norm,
        )

        self.stream_logs(logs)
        if self.sess_mgr:
            self.summarize_logs(logs)

    def stream_logs(self, logs):
        log_str = ""
        for key, params in self.col_params.items():
            abbr, width, precision = params.abbr, params.width, params.precision
            log_str += self.col_fmt.format(abbr=abbr, value=logs[key], width=width, precision=precision)
        log_str += "|"

        for stream in self.streams:
            print(log_str, file=stream)
            stream.flush()

    def summarize_logs(self, logs):
        losses = [logs['policy_loss'], logs['value_loss'], logs['entropy_loss']]
        rews = [logs['ep_rews_mean'], logs['ep_rews_std'], logs['ep_rews_max'], logs['ep_rews_min']]

        self.sess_mgr.add_summaries(['Mean', 'Std', 'Max', 'Min'], rews, 'Rewards', logs['updates'])
        self.sess_mgr.add_summaries(['Policy', 'Value', 'Entropy'], losses, 'Losses', logs['updates'])
        self.sess_mgr.add_summary('Grads Norm', logs['grads_norm'], 'Losses', logs['updates'])

    def on_start(self):
        self.start_time = time.time()
        if not self.log_file_path:
            return

        self.restore_logs()
        self.streams.append(open(self.log_file_path, 'a+'))

    def on_finish(self):
        if len(self.streams) > 1:
            self.streams[1].close()

    def restore_logs(self):
        if not os.path.isfile(self.log_file_path):
            return

        with open(self.log_file_path, 'r') as fl:
            last_line = fl.readlines()[-1]
        logs = last_line.split(" | ")
        self.run_time = int(logs[0].split(" ")[-1])
        self.env_eps.append(int(logs[2].split(" ")[-1]))


class AgentDebugLogger(Logger):
    def __init__(self, agent, log_freq=100, debug_steps=10):
        self.agent = agent
        self.log_freq = log_freq
        self.debug_steps = debug_steps

    def on_update(self, step, loss_terms, grads_norm, returns, adv, next_value):
        update_step = (step + 1) // self.agent.traj_len
        if update_step > 1 and update_step % self.log_freq:
            return

        np.set_printoptions(suppress=True, precision=2)
        n_steps = min(self.debug_steps, self.agent.traj_len)

        print()
        print("First Env For Last %d Steps:" % n_steps)
        print("Dones      ", self.agent.dones[-n_steps:, 0].flatten().astype(int))
        print("Rewards    ", self.agent.rewards[-n_steps:, 0].flatten())
        print("Values     ", self.agent.values[-n_steps:, 0].flatten(), round(next_value[0], 3))
        print("Returns    ", returns[-n_steps:, 0].flatten())
        print("Advs       ", adv[-n_steps:, 0].flatten())

        sys.stdout.flush()
