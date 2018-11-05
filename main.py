import reaver
import numpy as np
from absl import app
from absl import flags

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation.")
flags.DEFINE_integer("spatial_size", 16, "Resolution for spatial feature layers.")
FLAGS = flags.FLAGS


def main(argv):
    def act():
        function_id = np.random.choice(obs.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in env.act_spec().functions[function_id].args]
        return [function_id] + args

    env = reaver.env.SC2Env(FLAGS.map, FLAGS.spatial_size, FLAGS.step_mul, FLAGS.render)

    env.start()
    obs, rew, done = env.reset()
    for _ in range(1000):
        obs, rew, done = env.step(act())
        if done:
            obs, rew, done = env.reset()
    env.stop()


if __name__ == '__main__':
    app.run(main)
