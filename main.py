import reaver
import numpy as np
from absl import app
from absl import flags

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render first(!) env with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation.")
flags.DEFINE_integer("spatial_dim", 16, "Resolution for spatial feature layers.")
flags.DEFINE_integer("n_envs", 16, "Number of environments to execute in parallel.")
FLAGS = flags.FLAGS


def main(argv):
    FLAGS.n_envs = 1
    FLAGS.render = True

    def act():
        function_id = [np.random.choice(np.argwhere(obs[-1][i] > 0).flatten()) for i in range(FLAGS.n_envs)]
        args = [[[np.random.randint(0, size) for size in arg.shape] for _ in range(FLAGS.n_envs)]
                for arg in env.act_spec().spaces[1:]]
        return [function_id] + args

    env = reaver.env.MultiProcEnv(reaver.env.sc2_env.make_envs(FLAGS))
    env.start()
    obs, rew, done = env.reset()
    for _ in range(1000):
        obs, rew, done = env.step(act())
        if np.any(done):
            obs, rew, done = env.reset()
    env.stop()


if __name__ == '__main__':
    app.run(main)
