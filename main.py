import os
from absl import app
from absl import flags
import reaver as rvr

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render first(!) env with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation.")
flags.DEFINE_integer("spatial_dim", 16, "Resolution for spatial feature layers.")
flags.DEFINE_integer("envs", 4, "Number of environments to execute in parallel.")
flags.DEFINE_integer("steps", 2000, "Number of game steps to run (per environment).")
flags.DEFINE_string("gpu", "0", "id of GPU to use. If not set TensorFlow will default to CPU")


def main(argv):
    args = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import tensorflow as tf
    tf.reset_default_graph()

    env = rvr.envs.SC2Env(args.map, args.spatial_dim, args.step_mul, args.render)
    model = rvr.models.FullyConv(env.obs_spec(), env.act_spec())
    agent = rvr.agents.A2CAgent(model, args.envs)
    # agent = rvr.agents.RandomAgent(env.act_spec(), args.envs)
    agent.run(env, args.steps)


if __name__ == '__main__':
    # temp fix for annoying Cython bug
    # see https://github.com/cython/cython/issues/1720
    import warnings
    warnings.simplefilter('ignore', category=ImportWarning)

    app.run(main)
