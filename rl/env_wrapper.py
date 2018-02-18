from pysc2.lib.actions import FunctionCall, FUNCTIONS
from common.config import DEFAULT_ARGS, is_spatial


class EnvWrapper:
    def __init__(self, envs, config):
        self.envs, self.config = envs, config

    def step(self, acts):
        acts = self.wrap_actions(acts)
        results = self.envs.step(acts)
        return self.wrap_results(results)

    def reset(self):
        results = self.envs.reset()
        return self.wrap_results(results)

    def wrap_actions(self, actions):
        acts, args = actions[0], actions[1:]

        wrapped_actions = []
        for i, act in enumerate(acts):
            act_args = []
            for arg_type in FUNCTIONS[act].args:
                act_arg = [DEFAULT_ARGS[arg_type.name]]
                if arg_type.name in self.config.act_args:
                    act_arg = [args[self.config.arg_idx[arg_type.name]][i]]
                if is_spatial(arg_type.name):  # spatial args, convert to coords
                    act_arg = [act_arg[0] % self.config.sz, act_arg[0] // self.config.sz]  # (y,x), fix for PySC2
                act_args.append(act_arg)
            wrapped_actions.append(FunctionCall(act, act_args))

        return wrapped_actions

    def wrap_results(self, results):
        obs = [res.observation for res in results]
        rewards = [res.reward for res in results]
        dones = [res.last() for res in results]

        states = self.config.preprocess(obs)

        return states, rewards, dones

    def spec(self):
        return self.envs.spec()

    def close(self):
        return self.envs.close()

    @property
    def num_envs(self):
        return self.envs.num_envs