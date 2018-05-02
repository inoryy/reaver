import time
import numpy as np
from baselines import logger
from common import flatten_lists


class Runner:
    def __init__(self, envs, agent, n_steps=8):
        self.state = self.logs = self.ep_rews = None
        self.agent, self.envs, self.n_steps = agent, envs, n_steps

    def run(self, num_updates=1, train=True):
        # based on https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py
        self.reset()
        try:
            for i in range(num_updates):
                self.logs['updates'] += 1
                rollout = self.collect_rollout()
                if train:
                    self.agent.train(i, *rollout)
        except KeyboardInterrupt:
            pass
        finally:
            elapsed_time = time.time() - self.logs['start_time']
            frames = self.envs.num_envs * self.n_steps * self.logs['updates']
            print("Took %.3f seconds for %s steps: %.3f fps" % (elapsed_time, frames, frames / elapsed_time))

    def collect_rollout(self):
        states, actions = [None]*self.n_steps, [None]*self.n_steps
        rewards, dones, values = np.zeros((3, self.n_steps, self.envs.num_envs))

        for step in range(self.n_steps):
            action, values[step] = self.agent.act(self.state)
            states[step], actions[step] = self.state, action
            self.state, rewards[step], dones[step] = self.envs.step(action)

            self.log(rewards[step], dones[step])

        last_value = self.agent.get_value(self.state)

        return flatten_lists(states), flatten_lists(actions), rewards, dones, last_value, self.ep_rews

    def reset(self):
        self.state, *_ = self.envs.reset()
        self.logs = {'updates': 0, 'eps': 0, 'rew_best': 0, 'start_time': time.time(),
                     'ep_rew': np.zeros(self.envs.num_envs), 'dones': np.zeros(self.envs.num_envs)}

    def log(self, rewards, dones):
        self.logs['ep_rew'] += rewards
        self.logs['dones'] = np.maximum(self.logs['dones'], dones)
        if sum(self.logs['dones']) < self.envs.num_envs:
            return
        self.logs['eps'] += self.envs.num_envs
        self.logs['rew_best'] = max(self.logs['rew_best'], np.mean(self.logs['ep_rew']))

        elapsed_time = time.time() - self.logs['start_time']
        frames = self.envs.num_envs * self.n_steps * self.logs['updates']

        self.ep_rews = np.mean(self.logs['ep_rew'])
        logger.logkv('fps', int(frames / elapsed_time))
        logger.logkv('elapsed_time', int(elapsed_time))
        logger.logkv('n_eps', self.logs['eps'])
        logger.logkv('n_samples', frames)
        logger.logkv('n_updates', self.logs['updates'])
        logger.logkv('rew_best_mean', self.logs['rew_best'])
        logger.logkv('rew_max', np.max(self.logs['ep_rew']))
        logger.logkv('rew_mean', np.mean(self.logs['ep_rew']))
        logger.logkv('rew_mestd', np.std(self.logs['ep_rew'])) # weird name to ensure it's above min since logger sorts
        logger.logkv('rew_min', np.min(self.logs['ep_rew']))
        logger.dumpkvs()

        self.logs['dones'] = np.zeros(self.envs.num_envs)
        self.logs['ep_rew'] = np.zeros(self.envs.num_envs)
