import os
import reaver
from absl import app
from absl import flags
import tensorflow as tf

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render first(!) env with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation.")
flags.DEFINE_integer("spatial_dim", 16, "Resolution for spatial feature layers.")
flags.DEFINE_integer("envs", 4, "Number of environments to execute in parallel.")
flags.DEFINE_integer("steps", 2000, "Number of game steps to run (per environment).")
flags.DEFINE_string("gpu", "", "id of GPU to use. If not set TensorFlow will default to CPU")


def main(argv):
    args = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tf.reset_default_graph()
    sess = tf.Session()

    env = reaver.envs.SC2Env(args.map, args.spatial_dim, args.step_mul, args.render)
    model = reaver.models.FullyConv(env.obs_spec(), env.act_spec())
    # agent = reaver.agents.A2CAgent(policy, args.envs)
    # agent.run(env, args.steps)

    sess.run(tf.global_variables_initializer())
    import numpy as np
    obs = [np.random.randint(0, 5, size=(1,) + s.shape) for s in env.obs_spec().spaces]

    # TODO sampling
    out = sess.run(model.policy, feed_dict=dict(zip(model.inputs, obs)))
    print(", ".join(map(lambda o: str(o.shape), out)))


if __name__ == '__main__':
    app.run(main)
