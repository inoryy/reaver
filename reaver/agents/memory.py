import numpy as np
from . import RunningAgent


class MemoryAgent(RunningAgent):
    def __init__(self, base_shape, obs_spec, act_spec):
        """
        base_shape is not limited, but most common use case is (T, E)
        where T is number of time steps (batch size) and E is number of environments
        """
        RunningAgent.__init__(self)

        self.shape = base_shape
        self.batch_sz = self.shape[0]
        self.dones = np.empty(self.shape, dtype=np.bool)
        self.values = np.empty(self.shape, dtype=np.float32)
        self.rewards = np.empty(self.shape, dtype=np.float32)
        self.acts = [np.empty(self.shape + s.shape, dtype=s.dtype) for s in act_spec.spaces]
        self.obs = [np.empty(self.shape + s.shape, dtype=s.dtype) for s in obs_spec.spaces]

    def on_step(self, step, obs, action, reward, done, value=None):
        """
        Note: memory agent will overwrite previous batch without warning
        Keeping track of memory state is up to extending subclasses
        """
        step = step % self.batch_sz

        self.dones[step] = done
        self.rewards[step] = reward

        if value is not None:
            self.values[step] = value

        for i in range(len(obs)):
            self.obs[i][step] = obs[i]

        for i in range(len(action)):
            self.acts[i][step] = action[i]
