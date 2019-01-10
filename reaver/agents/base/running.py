import copy
from . import Agent
from reaver.envs.base import Env, MultiProcEnv


class RunningAgent(Agent):
    """
    Generic abstract class, defines API for interacting with an environment
    """
    def __init__(self):
        self.next_obs = None
        self.start_step = 0

    def run(self, env: Env, n_steps=1000000):
        env = self.wrap_env(env)
        env.start()
        try:
            self._run(env, n_steps)
        except KeyboardInterrupt:
            env.stop()
            self.on_finish()

    def _run(self, env, n_steps):
        self.on_start()
        obs, *_ = env.reset()
        obs = [o.copy() for o in obs]
        for step in range(self.start_step, self.start_step + n_steps):
            action, value = self.get_action_and_value(obs)
            self.next_obs, reward, done = env.step(action)
            self.on_step(step, obs, action, reward, done, value)
            obs = [o.copy() for o in self.next_obs]
        env.stop()
        self.on_finish()

    def get_action_and_value(self, obs):
        return self.get_action(obs), None

    def on_start(self): ...

    def on_step(self, step, obs, action, reward, done, value=None): ...

    def on_finish(self): ...

    def wrap_env(self, env: Env) -> Env:
        return env


class SyncRunningAgent(RunningAgent):
    """
    Abstract class that handles synchronous multiprocessing via MultiProcEnv helper
    Not meant to be used directly, extending classes automatically get the feature
    """
    def __init__(self, n_envs):
        RunningAgent.__init__(self)
        self.n_envs = n_envs

    def wrap_env(self, env: Env) -> Env:
        render, env.render = env.render, False
        envs = [env] + [copy.deepcopy(env) for _ in range(self.n_envs-1)]
        env.render = render

        return MultiProcEnv(envs)
