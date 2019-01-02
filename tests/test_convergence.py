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

        self.seed = 1234
        self.env = rvr.envs.GymEnv('CartPole-v0')

    def test_a2c(self):
        self._test_agent(rvr.agents.A2C)

    def test_ppo(self):
        self._test_agent(rvr.agents.PPO, n_epochs=3, minibatch_sz=128)

    def _test_agent(self, agent_cls, **kwargs):
        _kwargs = dict(optimizer=tf.train.AdamOptimizer(learning_rate=0.0007), entropy_coef=0.1, batch_sz=32,
                       gae_lambda=0.0, clip_grads_norm=0.0, clip_rewards=1.0, normalize_advantages=False, **kwargs)

        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0}))
        sess_mgr = rvr.utils.tensorflow.SessionManager(sess, base_path='/tmp/results/', checkpoint_freq=None)

        agent = agent_cls(self.env.obs_spec(), self.env.act_spec(), self._model_builder,
                          rvr.models.MultiPolicy, sess_mgr, n_envs=4, **_kwargs)
        agent.logger = rvr.utils.StreamLogger(n_envs=4, sess_mgr=sess_mgr)
        agent.logger.streams = []

        self.env._env.seed(self.seed)
        agent.run(self.env, 100 * agent.traj_len * agent.batch_sz // agent.n_envs)

        ep_rews = np.array(agent.logger.ep_rews_sum or [0])

        self.assertGreaterEqual(ep_rews.max(), 200.0)
        self.assertGreaterEqual(ep_rews.mean() + ep_rews.std(), 160.0)

    @staticmethod
    def _model_builder(obs_spec, act_spec):
        return rvr.models.build_mlp(obs_spec, act_spec, value_separate=True, obs_shift=True, obs_scale=True)


if __name__ == '__main__':
    unittest.main()
