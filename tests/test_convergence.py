import os
import warnings
import unittest
import numpy as np
import tensorflow as tf
import reaver as rvr


class TestConvergence(unittest.TestCase):
    def setUp(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        # mute annoying deprecation warning spam by TensorFlow
        # see https://github.com/tensorflow/tensorflow/issues/16152
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

        self.n_envs = 4
        self.batch_sz = 64
        self.n_updates = 100
        self.env = rvr.envs.GymEnv('CartPole-v0')

    def test_a2c(self):
        self._test_agent(rvr.agents.A2C)

    def test_ppo(self):
        self._test_agent(rvr.agents.PPO)

    def _test_agent(self, agent_cls):
        tf.reset_default_graph()

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0}))
        sess_mgr = rvr.utils.tensorflow.SessionManager(sess, base_path='/tmp/results/', checkpoint_freq=None)

        logger = rvr.utils.StreamLogger(n_envs=self.n_envs, sess_mgr=sess_mgr)
        logger.streams = []

        agent = agent_cls(
            self.env.obs_spec(), self.env.act_spec(), self._model_builder, rvr.models.MultiPolicy, sess_mgr,
            n_envs=self.n_envs, batch_sz=self.batch_sz, gae_lambda=0.0, clip_grads_norm=100.0, entropy_coef=0.0,
            value_coef=0.5, optimizer=tf.train.AdamOptimizer(learning_rate=0.005), logger=logger,
        )
        self.env._env.seed(0)
        agent.run(self.env, self.n_updates * agent.traj_len * agent.batch_sz // agent.n_envs)

        ep_rews = np.array(logger.ep_rews_sum or [0])

        self.assertGreaterEqual(ep_rews.max(), 200.0)
        self.assertGreaterEqual(ep_rews.mean() + ep_rews.std(), 180.0)

    @staticmethod
    def _model_builder(obs_spec, act_spec):
        return rvr.models.build_mlp(obs_spec, act_spec, activation='tanh', value_separate=True)


if __name__ == '__main__':
    unittest.main()
