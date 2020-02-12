from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box


class CartPole:
    """
    Wrapper for gym CartPole environment
    """

    def __init__(self, config=None):
        self.env = gym.make("CartPole-v0")
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space

    def reset(self):
        self.running_reward = 0
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return obs, score, done, info

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env), self.running_reward
