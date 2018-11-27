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
flags.DEFINE_integer('traj_len', None, 'Number of training samples to gather per one trajectory.')
flags.DEFINE_integer('batch_sz', None, 'Number of trajectories(!) to gather per one update.')
flags.DEFINE_integer('updates', 1000000, 'Number of train updates (1 update has batch_sz * traj_len samples).')

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
    'CartPole-v0': ['gym/base.gin'],

    'PongNoFrameskip-v0': ['atari/base.gin'],

    'InvertedPendulum-v2': ['mujoco/base.gin'],
    'HalfCheetah-v2':      ['mujoco/base.gin'],

    'MoveToBeacon':                ['sc2/move_to_beacon.gin'],
    'CollectMineralShards':        ['sc2/collect_mineral_shards.gin'],
    'DefeatRoaches':               ['sc2/defeat_roaches.gin'],
    'DefeatZerglingsAndBanelings': ['sc2/defeat_zerglings_and_banelings.gin'],
    'FindAndDefeatZerglings':      ['sc2/find_and_defeat_zerglings.gin'],
    'CollectMineralsAndGas':       ['sc2/collect_minerals_and_gas.gin'],
    'BuildMarines':                ['sc2/build_marines.gin'],
}


def main(argv):
    args = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.test:
        args.envs = 4
        args.batch_sz = 4
        args.log_freq = 10
        args.restore = True

    expt = rvr.utils.Experiment(args.results_dir, args.env, args.agent, args.experiment, args.restore)

    base_path = os.path.dirname(os.path.abspath(__file__))
    gin_files = gin_configs.get(args.env, ['base.gin'])
    gin_files = [os.path.join(base_path, 'configs', args.agent, fl) for fl in gin_files]
    if args.restore:
        gin_files += [expt.config_path]
    gin_files += args.gin_files

    if not args.gpu:
        args.gin_bindings.append("build_cnn_nature.data_format = 'channels_last'")
        args.gin_bindings.append("build_fully_conv.data_format = 'channels_last'")

    gin.parse_config_files_and_bindings(gin_files, args.gin_bindings)

    # TODO: do this the other way around - put these as gin bindings
    full_agent_name = str(agent_cls[args.agent])[:-2].split('.')[-1]

    if not args.traj_len:
        args.traj_len = int(gin.query_parameter(full_agent_name + '.traj_len'))

    if not args.batch_sz:
        args.batch_sz = int(gin.query_parameter(full_agent_name + '.batch_sz'))

    env_cls = rvr.envs.GymEnv if '-v' in args.env else rvr.envs.SC2Env
    env = env_cls(args.env, args.render, max_ep_len=args.max_ep_len)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess_mgr = rvr.utils.tensorflow.SessionManager(sess, expt.path, args.ckpt_freq, training_enabled=not args.test)

    agent = agent_cls[args.agent](env.obs_spec(), env.act_spec(), sess_mgr=sess_mgr,
                                  n_envs=args.envs, traj_len=args.traj_len, batch_sz=args.batch_sz)
    agent.logger = rvr.utils.StreamLogger(args.envs, args.log_freq, args.eps_avg, sess_mgr, expt.log_path)

    if sess_mgr.training_enabled:
        expt.save_gin_config(full_agent_name)
        expt.save_model_summary(agent.model)

    agent.run(env, args.updates * args.traj_len * args.batch_sz // args.envs)


if __name__ == '__main__':
    flags.mark_flag_as_required('env')
    app.run(main)
