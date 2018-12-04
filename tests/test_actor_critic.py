import unittest
import numpy as np
import reaver as rvr


class TestActorCritic(unittest.TestCase):
    def test_discounted_cumsum(self):
        discount = 0.99
        bootstrap = 5.0
        dones = np.array([0, 0, 0])
        rewards = np.array([1.0, 1.0, 1.0])

        discounts = discount * (1-dones)
        rewards = np.append(rewards, bootstrap)

        result = rvr.agents.A2C.discounted_cumsum(rewards, discounts)

        # 1.0 + 0.99*5.0 = 5.95
        # 1.0 + 0.99*1.0 + 0.99^2*5.0 = 6.8905
        # 1.0 + 0.99*1.0 + 0.99^2*1.0 + 0.99^3*5.0 = 7.821595
        expected = [7.821595, 6.8905, 5.95, 5.0]

        self.assertAlmostEqual(result.tolist(), expected)

    def test_discounted_cumsum_terminals(self):
        discount = 0.99
        bootstrap = 5.0
        dones = np.array([0, 1, 0])
        rewards = np.array([1.0, 1.0, 1.0])

        discounts = discount * (1-dones)
        rewards = np.append(rewards, bootstrap)

        result = rvr.agents.A2C.discounted_cumsum(rewards, discounts)

        # 1.0 + 0.99*5.0 = 5.95
        # 1.0 + 0 * (0.99*1.0 + 0.99^2*5.0) = 1.0
        # 1.0 + 0.99*1.0 + 0 * (0.99^2*1.0 + 0.99^3*5.0) = 1.99
        expected = [1.99, 1.0, 5.95, 5.0]

        self.assertAlmostEqual(result.tolist(), expected)
