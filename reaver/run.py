import os
import gin
import tensorflow as tf
from absl import app, flags

import reaver as rvr

flags.DEFINE_bool('restore', False,
                  'Restore & continue previously executed experiment. '
                  'If experiment not specified then last modified is used.')
flags.DEFINE_bool('test', False,
                  'Run an agent in test mode: restore flag is set to true and number of envs set to 1'
                  'Loss is calculated, but gradients are not applied.'
                  'Checkpoints, summaries, log files are not updated, but console logger is enabled.')

flags.DEFINE_string('env', None, 'Either Gym env id or PySC2 map name to run agent in.')
flags.DEFINE_string('agent', 'a2c', 'Name of the agent. Must be one of (a2c, ppo).')

flags.DEFINE_bool('render', False, 'Whether to render first(!) env.')
flags.DEFINE_string('gpu', '0', 'GPU(s) id(s) to use. If not set TensorFlow will use CPU.')

flags.DEFINE_integer('envs', 4, 'Number of environments to execute in parallel.')
flags.DEFINE_integer('batch_sz', None, 'Number of training samples to gather for 1 update.')
flags.DEFINE_integer('updates', 1000000, 'Number of train updates (1 update has batch_sz samples).')

flags.DEFINE_integer('ckpt_freq', 500, 'Number of train updates per one checkpoint save.')
flags.DEFINE_integer('log_freq', 100, 'Number of train updates per one console log.')
flags.DEFINE_integer('eps_avg', 100, 'Number of episodes to average for performance stats.')
flags.DEFINE_integer('max_ep_len', None, 'Max number of steps an agent can take in an episode.')

flags.DEFINE_string('results_dir', 'results', 'Directory for model weights, train logs, etc.')
flags.DEFINE_string('experiment', None, 'Name of the experiment. Datetime by default.')

flags.DEFINE_multi_string('gin_files', [], 'List of path(s) to gin config(s).')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings to override config values.')

agent_cls = {
    'a2c': rvr.agents.AdvantageActorCriticAgent,
    'ppo': rvr.agents.ProximalPolicyOptimizationAgent
}

gin_configs = {
    'CartPole-v0':          ['gym/base.gin'],

    'InvertedPendulum-v2':  ['mujoco/base.gin'],
    'HalfCheetah-v2':       ['mujoco/base.gin'],

    'PongNoFrameskip-v0':   ['atari/base.gin'],

    'DefeatRoaches':        ['sc2/base.gin'],
    'MoveToBeacon':         ['sc2/move_to_beacon.gin'],
    'CollectMineralShards': ['sc2/collect_mineral_shards.gin'],
}


def main(argv):
    args = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.test:
        args.envs = 1
        args.restore = True
    expt = rvr.utils.Experiment(args.results_dir, args.env, args.agent, args.experiment, args.restore)

    gin_files = gin_configs.get(args.env, [])
    gin_files = ['reaver/configs/' + fl for fl in gin_files]
    if args.restore:
        gin_files += [expt.config_path]
    gin_files += args.gin_files

    if not args.gpu:
        args.gin_bindings.append("build_cnn_nature.data_format = 'channels_last'")
        args.gin_bindings.append("build_fully_conv.data_format = 'channels_last'")

    gin.parse_config_files_and_bindings(gin_files, args.gin_bindings)

    if not args.batch_sz:
        args.batch_sz = int(gin.query_parameter('ActorCriticAgent.batch_sz'))

    env_cls = rvr.envs.GymEnv if '-v' in args.env else rvr.envs.SC2Env
    env = env_cls(args.env, args.render, max_ep_len=args.max_ep_len)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess_mgr = rvr.utils.tensorflow.SessionManager(sess, expt.path, args.ckpt_freq, training_enabled=not args.test)

    agent = agent_cls[args.agent](sess_mgr, env.obs_spec(), env.act_spec(), args.envs, args.batch_sz)
    agent.logger = rvr.utils.StreamLogger(args.envs, agent.traj_len, args.log_freq, args.eps_avg, sess_mgr, expt.log_path)

    if sess_mgr.training_enabled:
        expt.save_gin_config()
        expt.save_model_summary(agent.model)

    agent.run(env, args.updates * args.batch_sz // args.envs)


if __name__ == '__main__':
    flags.mark_flag_as_required('env')
    app.run(main)
