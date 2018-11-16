import os
import gin
import tensorflow as tf
from absl import app, flags

import reaver as rvr

flags.DEFINE_string('env', None, 'Either Gym env id or PySC2 map name to run agent in.')
flags.DEFINE_string('agent', 'a2c', 'Name of the agent. Must be one of (a2c, ppo).')
flags.DEFINE_integer('envs', 4, 'Number of environments to execute in parallel.')
flags.DEFINE_integer('traj_len', None, 'Length of the trajectory an agent takes before training.')
flags.DEFINE_integer('updates', 100, 'Number of train updates (1 update has traj_len*envs samples).')
flags.DEFINE_string('data_dir', 'data', 'Data directory for model weights, train logs, etc.')
flags.DEFINE_string('gpu', '0', 'GPU(s) id(s) to use. If not set TensorFlow will use CPU.')
flags.DEFINE_bool('render', False, 'Whether to render first(!) env.')
flags.DEFINE_multi_string('gin_files', [], 'List of path(s) to gin config(s).')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings to override config values.')

agent_cls = {
    'a2c': rvr.agents.AdvantageActorCriticAgent,
    'ppo': rvr.agents.ProximalPolicyOptimizationAgent
}

gin_configs = {
    'gym_control.gin': ['CartPole-v0', 'Pendulum-v0']
}


def main(argv):
    args = flags.FLAGS

    gin_files = ['reaver/configs/' + k for k, v in gin_configs.items() if args.env in v]
    gin_files += args.gin_files
    gin.parse_config_files_and_bindings(gin_files, args.gin_bindings)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    traj_len = args.traj_len if args.traj_len else int(gin.query_parameter('ActorCriticAgent.traj_len'))
    env_cls = rvr.envs.GymEnv if '-v' in args.env else rvr.envs.SC2Env

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    env = env_cls(args.env, args.render)
    agent = agent_cls[args.agent](sess, env.obs_spec(), env.act_spec(), args.envs, traj_len)

    agent.run(env, args.updates * traj_len)


if __name__ == '__main__':
    flags.mark_flag_as_required('env')
    app.run(main)
