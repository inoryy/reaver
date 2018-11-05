import reaver
from absl import app
from absl import flags

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render first(!) env with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation.")
flags.DEFINE_integer("spatial_dim", 16, "Resolution for spatial feature layers.")
flags.DEFINE_integer("n_envs", 4, "Number of environments to execute in parallel.")


def main(argv):
    args = flags.FLAGS
    env = reaver.env.SC2Env(args.map, args.spatial_dim, args.step_mul, args.render)
    agent = reaver.agent.RandomAgent(env.obs_spec(), env.act_spec(), args.n_envs)
    agent.run(env, 1000)


if __name__ == '__main__':
    app.run(main)
