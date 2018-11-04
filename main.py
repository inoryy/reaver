import numpy as np
from absl import flags
from env import SC2Env

if __name__ == '__main__':
    flags.FLAGS(['main.py'])

    def act():
        function_id = np.random.choice(obs.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in env.act_spec().functions[function_id].args]
        return [function_id] + args

    env = SC2Env(render=True)
    env.start()
    obs, rew, done = env.reset()
    tot_rew = 0
    for _ in range(1000):
        obs, rew, done = env.step(act())
        tot_rew += rew
        if done:
            print(tot_rew)
            tot_rew = 0
            obs, rew, done = env.reset()
    env.stop()
