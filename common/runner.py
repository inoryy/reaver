import time
from common.env import make_env, EnvPool


# based on https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py
class Runner(object):
    def __init__(self, n_envs, map_name, **env_params):
        self.n_envs = n_envs
        self.map_name = map_name
        self.env_params = env_params

    def run(self, agent, max_frames=0):
        envs = EnvPool([make_env(self.map_name, **self.env_params) for _ in range(self.n_envs)])
        agent.setup(*envs.spec())

        total_frames = 0
        start_time = time.time()

        try:
            obs = envs.reset()
            while True:
                total_frames += self.n_envs
                if max_frames and total_frames >= self.n_envs * max_frames:
                    break
                acts = agent.step(obs)
                obs = envs.step(acts)
        except KeyboardInterrupt:
            pass
        finally:
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))
            envs.close()
