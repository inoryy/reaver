from abc import abstractmethod
from . import Agent


class RunningAgent(Agent):
    def run(self, env, n_steps=1000000):
        env = self.wrap_env(env)
        env.start()
        try:
            self._run(env, n_steps)
        except KeyboardInterrupt:
            env.stop()

    def _run(self, env, n_steps):
        obs, *_ = env.reset()
        for n_step in range(1, n_steps+1):
            action = self.get_action(obs)
            next_obs, reward, done = env.step(action)
            obs = next_obs
        env.stop()

    @abstractmethod
    def wrap_env(self, env): ...


class SyncRunningAgent(RunningAgent):
    def __init__(self, obs_spec, act_spec, n_envs=2):
        super().__init__(obs_spec, act_spec)
        self.n_envs = n_envs

    def wrap_env(self, env):
        import copy
        import reaver

        render, env.render = env.render, False
        envs = [env] + [copy.deepcopy(env) for _ in range(self.n_envs-1)]
        env.render = render

        return reaver.env.MultiProcEnv(envs)
