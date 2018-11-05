import reaver
import numpy as np
from absl import app
from absl import flags

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation.")
flags.DEFINE_integer("spatial_dim", 16, "Resolution for spatial feature layers.")
FLAGS = flags.FLAGS


def main(argv):
    def act():
        function_id = np.random.choice(np.argwhere(obs[-1] > 0).flatten())
        args = [[np.random.randint(0, size) for size in arg.shape]
                for arg in env.act_spec().spaces[1:]]
        return [function_id] + args

    env = reaver.env.SC2Env(FLAGS.map, FLAGS.spatial_dim, FLAGS.step_mul, FLAGS.render)
    env.start()
    obs, rew, done = env.reset()
    for _ in range(1000):
        obs, rew, done = env.step(act())
        if done:
            obs, rew, done = env.reset()
    env.stop()


if __name__ == '__main__':
    app.run(main)
