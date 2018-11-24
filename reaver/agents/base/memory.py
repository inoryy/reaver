import numpy as np
from .running import RunningAgent


class MemoryAgent(RunningAgent):
    def __init__(self, obs_spec, act_spec, traj_len, batch_sz):
        RunningAgent.__init__(self)

        self.traj_len = traj_len
        self.batch_sz = batch_sz
        self.shape = (traj_len, batch_sz)
        self.batch_ptr = 0
        self.n_batches = 0

        self.dones = np.empty(self.shape, dtype=np.bool)
        self.values = np.empty(self.shape, dtype=np.float32)
        self.rewards = np.empty(self.shape, dtype=np.float32)
        self.acts = [np.empty(self.shape + s.shape, dtype=s.dtype) for s in act_spec.spaces]
        self.obs = [np.empty(self.shape + s.shape, dtype=s.dtype) for s in obs_spec.spaces]
        self.last_obs = [np.empty((self.batch_sz, ) + s.shape, dtype=s.dtype) for s in obs_spec.spaces]

    def on_step(self, step, obs, action, reward, done, value=None):
        """
        Note: memory agent will overwrite previous batch without warning
        Keeping track of memory state is up to extending subclasses
        """
        step = step % self.traj_len
        self.batch_ptr = self.batch_ptr % self.batch_sz
        bs, be = self.batch_ptr, self.batch_ptr + reward.shape[0]

        self.dones[step, bs:be] = done
        self.rewards[step, bs:be] = reward

        if value is not None:
            self.values[step, bs:be] = value

        for i in range(len(obs)):
            self.obs[i][step, bs:be] = obs[i]

        for i in range(len(action)):
            self.acts[i][step, bs:be] = action[i]

        if (step+1) % self.traj_len == 0:
            # finished one trajectory
            for i in range(len(obs)):
                self.last_obs[i][bs:be] = self.next_obs[i]
            self.batch_ptr += reward.shape[0]

        if self.batch_ready():
            self.n_batches += 1

    def batch_ready(self):
        return self.batch_ptr == self.batch_sz
