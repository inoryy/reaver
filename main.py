import os
import reaver as rvr
from absl import app, flags

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render first(!) env with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation.")
flags.DEFINE_integer("spatial_dim", 16, "Resolution for spatial feature layers.")
flags.DEFINE_integer("envs", 4, "Number of environments to execute in parallel.")
flags.DEFINE_integer("batch", 16, "Steps agent takes in-between training procedure.")
flags.DEFINE_integer("steps", 2000, "Number of game steps to run (per environment).")
flags.DEFINE_string("gpu", "0", "id(s) of GPU(s) to use. If not set TensorFlow will default to CPU")


def main(argv):
    args = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    env = rvr.envs.SC2Env(args.map, args.spatial_dim, args.step_mul, args.render)
    agent = rvr.agents.A2CAgent(rvr.models.FullyConv, env.obs_spec(), env.act_spec(), args.envs, args.batch)
    # agent = rvr.agents.RandomAgent(env.act_spec(), args.envs)
    agent.run(env, args.steps)


if __name__ == '__main__':
    # temp fix for annoying Cython bug
    # see https://github.com/cython/cython/issues/1720
    import warnings
    warnings.simplefilter('ignore', category=ImportWarning)

    app.run(main)
