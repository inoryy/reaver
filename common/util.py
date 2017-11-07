import numpy as np


class RolloutStorage(object):
    def __init__(self):
        self.states = None
        self.screen_cat_states = []
        self.screen_num_states = []
        self.minimap_cat_states = []
        self.minimap_num_states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.dones = []

    def insert(self, state, action, reward, value):
        screen_cat, screen_num, minimap_cat, minimap_num = state
        self.screen_cat_states.append(screen_cat)
        self.screen_num_states.append(screen_num)
        self.minimap_cat_states.append(minimap_cat)
        self.minimap_num_states.append(minimap_num)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.returns.append([0]*action.shape[0])
        # TODO track done flags
        self.dones.append([0]*action.shape[0])

    def flatten(self):
        screen_cat_states = np.concatenate(self.screen_cat_states, axis=0)
        screen_num_states = np.concatenate(self.screen_num_states, axis=0)
        minimap_cat_states = np.concatenate(self.minimap_cat_states, axis=0)
        minimap_num_states = np.concatenate(self.minimap_num_states, axis=0)
        self.states = screen_cat_states, screen_num_states, minimap_cat_states, minimap_num_states
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.values = np.concatenate(self.values, axis=0)
        self.returns = np.concatenate(self.returns, axis=0)
        self.dones = np.concatenate(self.dones, axis=0)

    # TODO GAE
    def compute_returns(self, last_value, gamma):
        self.rewards = self.rewards[1:]
        self.returns.append(last_value)
        self.dones = np.array(self.dones)
        self.rewards = np.array(self.rewards)
        self.returns = np.array(self.returns)
        for step in reversed(range(self.rewards.shape[0])):
            self.returns[step] = self.returns[step+1] * gamma * (1 - self.dones[step]) + self.rewards[step]
        self.returns = self.returns[:-1]
        self.flatten()

    def inputs(self):
        return self.states + (self.returns, )
