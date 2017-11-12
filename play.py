import sys, importlib
from absl import flags
from common.runner import Runner

FLAGS = flags.FLAGS
flags.DEFINE_string("map_name", "MoveToBeacon", "Which map to use.")
flags.DEFINE_string("agent", "agent.rl.RLAgent", "Which agent to run.")
flags.DEFINE_integer("n_envs", 10, "Number of SC2 environments to run in parallel.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("max_steps", 2000, "Max steps per env.")

if __name__ == '__main__':
    FLAGS(sys.argv)

    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    agent_args = []
    if agent_name == 'RLAgent':
        import tensorflow as tf
        tf.reset_default_graph()
        sess = tf.Session()
        feats = {'screen': ['player_relative', 'unit_hit_points'], 'minimap': ['player_relative', 'height_map']}
        agent_args = [sess, feats]

    agent = agent_cls(*agent_args)
    runner = Runner(FLAGS.n_envs, FLAGS.map_name, visualize=FLAGS.render)
    runner.run(agent, FLAGS.max_steps)