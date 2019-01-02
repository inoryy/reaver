import numpy as np
from . import Env, Spec, Space
from reaver.envs.atari import AtariPreprocessing


class GymEnv(Env):
    def __init__(self, _id='CartPole-v0', render=False, reset_done=True, max_ep_len=None):
        super().__init__(_id, render, reset_done, max_ep_len)

        self._env = None
        self.specs = None
        self.ep_step = 0

    def start(self):
        import gym  # lazy-loading
        gym.logger.set_level(40)  # avoid annoying internal warn messages

        self._env = gym.make(self.id)

        try:
            import atari_py
        except ImportError:
            return

        if any([env_name in self.id.lower() for env_name in atari_py.list_games()]):
            self._env = AtariPreprocessing(self._env.env)

        self.make_specs(running=True)

    def step(self, action):
        obs, reward, done, _ = self._env.step(self.wrap_act(action))

        self.ep_step += 1
        if self.ep_step >= self.max_ep_len:
            done = 1

        if done and self.reset_done:
            obs = self.reset(wrap=False)

        obs = self.wrap_obs(obs)

        if self.render:
            self._env.render()

        # TODO what if reward is a float?
        return obs, int(reward), int(done)

    def reset(self, wrap=True):
        obs = self._env.reset()

        if wrap:
            obs = self.wrap_obs(obs)

        if self.render:
            self._env.render()

        self.ep_step = 0

        return obs

    def stop(self):
        self._env.close()

    def wrap_act(self, act):
        if len(self.act_spec().spaces) == 1:
            act = act[0]
        return act

    def wrap_obs(self, obs):
        if len(self.obs_spec().spaces) == 1:
            obs = [obs]
        # can't trust gym space definitions it seems...
        obs = [ob.astype(sp.dtype) for ob, sp in zip(obs, self.obs_spec().spaces)]
        return obs

    def obs_spec(self):
        if not self.specs:
            self.make_specs()
        return self.specs['obs']

    def act_spec(self):
        if not self.specs:
            self.make_specs()
        return self.specs['act']

    def make_specs(self, running=False):
        render, self.render = self.render, False
        if not running:
            self.start()
        self.specs = {
            'obs': Spec(parse(self._env.observation_space), 'Observation'),
            'act': Spec(parse(self._env.action_space), 'Action')
        }
        if not running:
            self.stop()
        self.render = render


def parse(gym_space, name=None):
    cls_name = type(gym_space).__name__

    if cls_name == 'Discrete':
        return [Space(dtype=gym_space.dtype, domain=(0, gym_space.n), categorical=True, name=name)]

    if cls_name == 'Box':
        lo, hi = gym_space.low, gym_space.high
        return [Space(shape=gym_space.shape, dtype=gym_space.dtype, domain=(lo, hi), name=name)]

    if cls_name == 'Tuple':
        spaces = []
        for space in gym_space.spaces:
            spaces += parse(space)
        return spaces

    if cls_name == "Dict":
        spaces = []
        for name, space in gym_space.spaces.items():
            spaces += parse(space, name)
        return spaces
