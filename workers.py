'''import collections
import cv2
import numpy as np
import gym


class RepeatAction(gym.Wrapper):
    def __init__(self, env=None, repeat=4, fire_first= False):
        super(RepeatAction, self).__init__()
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first

    def step(step, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info ='''


