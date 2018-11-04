from absl import flags
from env import SC2Env

if __name__ == '__main__':
    flags.FLAGS(['main.py'])

    env = SC2Env(render=True)
    env.start()
    env.reset()

    import time
    time.sleep(5)
    print(env.obs_spec())
    print(env.act_spec())
    env.stop()
