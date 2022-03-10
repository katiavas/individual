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


import collections
import cv2
import numpy as np
import gym

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape,
                                                dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs


def make_env(env_name, shape=(84, 84, 1)):
    env = gym.make(env_name)
    env = PreprocessFrame(shape, env)
    return env