from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box


class SwingUpCartPole(gym.Env):
    """
    Wrapper for gym CartPole environment
    """

    def __init__(self, config=None):
        # Gym properties
        self.env = gym.make("CartPole-v0")
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space

        # Time limit
        self.t = 0
        self.max_t = 200

        # SwingUp properties
        self.theta_dot_threshold = 4*np.pi
        self.theta_threshold_radians = deepcopy(self.env.theta_threshold_radians)
        self.done = False
        self.env.reset()
    
    def reset(self):
        self.env.state = [0, 0, np.pi, 0] + self.env.reset()
        #self.state = [0,0,np.pi,0] + self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.env.steps_beyond_done = None
        return np.array(self.env.state)
    
    def render(self):
        return self.env.render()
    
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.done = done
        x, x_dot, theta, theta_dot = obs
        
        done = x < -self.env.x_threshold \
               or x > self.env.x_threshold \
               or theta_dot < -self.theta_dot_threshold \
               or theta_dot > self.theta_dot_threshold
        
        if done:
            # game over
            reward = -10.
            if self.env.steps_beyond_done is None:
                self.env.steps_beyond_done = 0
            else:
                print('wtf')
                self.env.steps_beyond_done += 1
        else:
            if -self.theta_threshold_radians < theta and theta < self.theta_threshold_radians:
                # pole upright
                reward = 1.
            else:
                # pole swinging
                reward = 0.

        return np.array(obs), reward, done, {}

    def set_state(self, state):
        self.env.env.state = deepcopy(state[0])
        self.t = deepcopy(state[1])
        self.done = deepcopy(state[2])
        self.env.steps_beyond_done = deepcopy(state[3])
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy((self.env.env.state, self.t, self.done, self.env.steps_beyond_done))