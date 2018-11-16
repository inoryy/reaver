import os
import gin
import tensorflow as tf
from absl import app, flags

import reaver as rvr

flags.DEFINE_bool('render', False, 'Whether to render first(!) env.')
flags.DEFINE_string('env', None, 'Either Gym env id or PySC2 map name to run agent in.')
flags.DEFINE_string('agent', 'a2c', 'Name of the agent. Must be one of (a2c, ppo).')
flags.DEFINE_integer('envs', 4, 'Number of environments to execute in parallel.')
flags.DEFINE_integer('batch_sz', None, 'Number of training samples to gather for 1 update.')
flags.DEFINE_integer('updates', 100, 'Number of train updates (1 update has batch_sz samples).')
flags.DEFINE_integer('max_ep_len', None, 'Max number of steps an agent can take in an episode.')
flags.DEFINE_string('data_dir', 'data', 'Data directory for model weights, train logs, etc.')
flags.DEFINE_string('gpu', '0', 'GPU(s) id(s) to use. If not set TensorFlow will use CPU.')
flags.DEFINE_multi_string('gin_files', [], 'List of path(s) to gin config(s).')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings to override config values.')

agent_cls = {
    'a2c': rvr.agents.AdvantageActorCriticAgent,
    'ppo': rvr.agents.ProximalPolicyOptimizationAgent
}

gin_configs = {
    'CartPole-v0':         ['gym_classic.gin'],
    'InvertedPendulum-v2': ['gym_classic.gin', 'gym_continuous.gin'],
    'HalfCheetah-v2':      ['gym_classic.gin', 'gym_continuous.gin', 'mujoco.gin']
}


def main(argv):
    args = flags.FLAGS

    gin_files = gin_configs.get(args.env, [])
    gin_files = ['reaver/configs/' + fl for fl in gin_files]
    gin_files += args.gin_files
    gin.parse_config_files_and_bindings(gin_files, args.gin_bindings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    batch_sz = args.batch_sz if args.batch_sz else int(gin.query_parameter('ActorCriticAgent.batch_sz'))
    env_cls = rvr.envs.GymEnv if '-v' in args.env else rvr.envs.SC2Env

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    env = env_cls(args.env, args.render, max_ep_len=args.max_ep_len)
    agent = agent_cls[args.agent](sess, env.obs_spec(), env.act_spec(), args.envs, batch_sz)

    agent.run(env, args.updates * batch_sz // args.envs)


if __name__ == '__main__':
    flags.mark_flag_as_required('env')
    app.run(main)
