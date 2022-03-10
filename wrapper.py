import cv2
import numpy as np
import gym

# Wrapper that helps modify/preprocess observations
class InputImg(gym.ObservationWrapper):
    def __init__(self, input_shape, env=None):
        # call super constructor with the environment as an input
        super(InputImg, self).__init__(env)
        self.shape = (input_shape[2], input_shape[0], input_shape[1])
        """" Extract space dimensions """
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape,
                                                dtype=np.float32)
        print(self.observation_space)

    def observation(self, obs):
        input_img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        img_resised = cv2.resize(input_img, self.shape[1:], interpolation=cv2.INTER_AREA)  # self.shape[1:] = 84,84
        # self.shape will be either 1 for grayscale or 3 for coloured image
        new_obs = np.array(img_resised, dtype=np.uint8).reshape((1, 84, 84))
        # make pixel values between 0 and 1
        new_obs = new_obs / 255.0
        return new_obs


def make_env(env_name, shape=(84, 84, 1)):
    env = gym.make(env_name)
    env = InputImg(shape, env)
    return env
