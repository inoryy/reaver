import reaver
from absl import app
from absl import flags

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render first(!) env with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation.")
flags.DEFINE_integer("spatial_dim", 16, "Resolution for spatial feature layers.")
flags.DEFINE_integer("envs", 4, "Number of environments to execute in parallel.")
flags.DEFINE_integer("steps", 2000, "Number of game steps to run (per environment).")


def main(argv):
    args = flags.FLAGS
    env = reaver.envs.SC2Env(args.map, args.spatial_dim, args.step_mul, args.render)
    agent = reaver.agents.RandomAgent(env.obs_spec(), env.act_spec(), args.envs)
    agent.run(env, args.steps)


if __name__ == '__main__':
    app.run(main)
