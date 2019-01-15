import os
import gin
import tensorflow as tf
from absl import app, flags

import reaver as rvr

flags.DEFINE_string('env', None, 'Either Gym env id or PySC2 map name to run agent in.')
flags.DEFINE_string('agent', 'a2c', 'Name of the agent. Must be one of (a2c, ppo).')

flags.DEFINE_bool('render', False, 'Whether to render first(!) env.')
flags.DEFINE_string('gpu', '0', 'GPU(s) id(s) to use. If not set TensorFlow will use CPU.')

flags.DEFINE_integer('n_envs', 4, 'Number of environments to execute in parallel.')
flags.DEFINE_integer('n_updates', 1000000, 'Number of train updates (1 update has batch_sz * traj_len samples).')

flags.DEFINE_integer('ckpt_freq', 500, 'Number of train updates per one checkpoint save.')
flags.DEFINE_integer('log_freq', 100, 'Number of train updates per one console log.')
flags.DEFINE_integer('log_eps_avg', 100, 'Number of episodes to average for performance stats.')
flags.DEFINE_integer('max_ep_len', None, 'Max number of steps an agent can take in an episode.')

flags.DEFINE_string('results_dir', 'results', 'Directory for model weights, train logs, etc.')
flags.DEFINE_string('experiment', None, 'Name of the experiment. Datetime by default.')

flags.DEFINE_multi_string('gin_files', [], 'List of path(s) to gin config(s).')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings to override config values.')

flags.DEFINE_bool('restore', False,
                  'Restore & continue previously executed experiment. '
                  'If experiment not specified then last modified is used.')
flags.DEFINE_bool('test', False,
                  'Run an agent in test mode: restore flag is set to true and number of envs set to 1'
                  'Loss is calculated, but gradients are not applied.'
                  'Checkpoints, summaries, log files are not updated, but console logger is enabled.')

flags.DEFINE_alias('e', 'env')
flags.DEFINE_alias('a', 'agent')
flags.DEFINE_alias('p', 'n_envs')
flags.DEFINE_alias('u', 'n_updates')
flags.DEFINE_alias('lf', 'log_freq')
flags.DEFINE_alias('cf', 'ckpt_freq')
flags.DEFINE_alias('la', 'log_eps_avg')
flags.DEFINE_alias('n', 'experiment')
flags.DEFINE_alias('g', 'gin_bindings')


def main(argv):
    args = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.env in rvr.utils.config.SC2_MINIGAMES_ALIASES:
        args.env = rvr.utils.config.SC2_MINIGAMES_ALIASES[args.env]

    if args.test:
        args.n_envs = 1
        args.log_freq = 1
        args.restore = True

    expt = rvr.utils.Experiment(args.results_dir, args.env, args.agent, args.experiment, args.restore)

    gin_files = rvr.utils.find_configs(args.env, os.path.dirname(os.path.abspath(__file__)))
    if args.restore:
        gin_files += [expt.config_path]
    gin_files += args.gin_files

    if not args.gpu:
        args.gin_bindings.append("build_cnn_nature.data_format = 'channels_last'")
        args.gin_bindings.append("build_fully_conv.data_format = 'channels_last'")

    gin.parse_config_files_and_bindings(gin_files, args.gin_bindings)
    args.n_envs = min(args.n_envs, gin.query_parameter('ACAgent.batch_sz'))

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess_mgr = rvr.utils.tensorflow.SessionManager(sess, expt.path, args.ckpt_freq, training_enabled=not args.test)

    env_cls = rvr.envs.GymEnv if '-v' in args.env else rvr.envs.SC2Env
    env = env_cls(args.env, args.render, max_ep_len=args.max_ep_len)

    agent = rvr.agents.registry[args.agent](env.obs_spec(), env.act_spec(), sess_mgr=sess_mgr, n_envs=args.n_envs)
    agent.logger = rvr.utils.StreamLogger(args.n_envs, args.log_freq, args.log_eps_avg, sess_mgr, expt.log_path)

    if sess_mgr.training_enabled:
        expt.save_gin_config()
        expt.save_model_summary(agent.model)

    agent.run(env, args.n_updates * agent.traj_len * agent.batch_sz // args.n_envs)


if __name__ == '__main__':
    flags.mark_flag_as_required('env')
    app.run(main)
