from .abc_env import Env
from pysc2.lib.actions import FunctionCall, FUNCTIONS


class SC2Env(Env):
    def __init__(self, map_name='MoveToBeacon', size=16, render=False):
        self._env = None
        self.map_name, self.sz, self.render = map_name, size, render

    def start(self):
        from pysc2.env import sc2_env
        self._env = sc2_env.SC2Env(
            map_name=self.map_name,
            visualize=self.render,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=self.sz,
                feature_minimap=self.sz,
                rgb_screen=None,
                rgb_minimap=None,
                use_feature_units=False
            ),
            step_mul=8,
        )

    def step(self, action):
        pass

    def reset(self):
        return self._env.reset()

    def stop(self):
        self._env.close()

    def obs_spec(self):
        return self._env.observation_spec()

    def act_spec(self):
        return self._env.action_spec()
