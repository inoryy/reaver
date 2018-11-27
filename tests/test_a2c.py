import os
import warnings
import unittest
import numpy as np
import tensorflow as tf
import reaver as rvr


class TestA2C(unittest.TestCase):
    def setUp(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        # mute annoying deprecation warning spam by TensorFlow
        # see https://github.com/tensorflow/tensorflow/issues/16152
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

    def test_convergence(self):
        n_envs = 4
        batch_sz = 32
        n_updates = 100

        env = rvr.envs.GymEnv('CartPole-v0')
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0}))
        sess_mgr = rvr.utils.tensorflow.SessionManager(sess, base_path='/tmp/results', checkpoint_freq=-1)

        logger = rvr.utils.StreamLogger(n_envs=n_envs, sess_mgr=sess_mgr)
        logger.streams = []

        agent = rvr.agents.A2C(
            env.obs_spec(), env.act_spec(), self._model_builder, rvr.models.MultiPolicy, sess_mgr, n_envs,
            batch_sz=batch_sz, gae_lambda=0.0, clip_grads_norm=1.0, entropy_coef=0.01,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.005), logger=logger,
        )

        agent.run(env, n_updates * agent.traj_len * agent.batch_sz // agent.n_envs)
        ep_rews = np.array(logger.ep_rews_sum or [0])

        self.assertGreaterEqual(ep_rews.mean(), 100.0)
        self.assertGreaterEqual(ep_rews.mean() + ep_rews.std(), 150.0)
        self.assertGreaterEqual(ep_rews.max(), 190.0)

    @staticmethod
    def _model_builder(obs_spec, act_spec):
        return rvr.models.build_mlp(obs_spec, act_spec, activation='tanh', value_separate=True)


if __name__ == '__main__':
    unittest.main()
