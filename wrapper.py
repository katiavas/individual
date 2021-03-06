import cv2
import numpy as np
import gym
import collections
''''
class RepeatAction(gym.Wrapper):
    def __init__(self, env=None, repeat=4, fire_first=False):
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first
    # set the total reward to 0 and done to False
    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env_step(1)
        return obs
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


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                env.observation_space.low.repeat(repeat, axis=0),
                env.observation_space.high.repeat(repeat, axis=0),
                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(42, 42, 1), repeat=4):
    env = gym.make(env_name)
    # env = RepeatAction(env, repeat)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    return env
'''


# https://alexandervandekleut.github.io/gym-wrappers/
# Wrapper that helps modify/preprocess observations
class InputImg(gym.ObservationWrapper):
    def __init__(self, input_shape, env):
        # call super constructor with the environment as an input
        super(InputImg, self).__init__(env)
        self.shape = (input_shape[2], input_shape[0], input_shape[1])
        """" Extract space dimensions """
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape,
                                                dtype=np.float32)

    # override the observation method of the environment
    def observation(self, obs):
        input_img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        img_resised = cv2.resize(input_img, self.shape[1:], interpolation=cv2.INTER_AREA)  # self.shape[1:] = 84,84
        # self.shape will be either 1 for grayscale or 3 for coloured image
        new_obs = np.array(img_resised, dtype=np.uint8).reshape((1, 42, 42))
        # make pixel values between 0 and 1
        new_obs = new_obs / 255.0
        return new_obs


def make_env(env_name, shape=(42, 42, 1)):
    env = gym.make(env_name)
    env = InputImg(shape, env)
    return env
