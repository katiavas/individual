import cv2
import numpy as np
import gym


class InputImg(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        # call super constructor with the environment as an input
        super(InputImg, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape,
                                                dtype=np.float32)
        print(self.observation_space)

    # self.shape[1:] = 84,84
    def observation(self, obs):
        input_img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        img_resised = cv2.resize(input_img, self.shape[1:], interpolation=cv2.INTER_AREA)
        # self.shape will be either 1 for grayscale or 3 for coloured image
        new_obs = np.array(img_resised, dtype=np.uint8).reshape((1, 84, 84))
        # make pixel values between 0 and 1
        new_obs = new_obs / 255.0
        return new_obs


def make_env(env_name, shape=(84, 84, 1)):
    env = gym.make(env_name)
    env = InputImg(shape, env)
    return env
