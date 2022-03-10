import gym
import numpy as np
import torch as T
from actor_critic import ActorCritic
from icm import ICM
from memory import Memory
from utils import plot_learning_curve
# from utils import plot_intrinsic_reward
import torch as T
import cv2
# from utils import plot_learning_curve_with_shaded_error
import matplotlib.pyplot as plt
from wrapper import make_env

from skimage.transform import resize

'''
def downscale_obs(obs, new_size=(42, 42), to_gray=True):
    if to_gray:
        return resize(obs, new_size, anti_aliasing=True).max(axis=2)
    else:
        return resize(obs, new_size, anti_aliasing=True)


def prepare_state(state):
    return T.from_numpy(downscale_obs(state, to_gray=True)).float().unsqueeze(dim=0)


def prepare_multi_state(state1, state2):
    state1 = state1.clone()
    tmp = T.from_numpy(downscale_obs(state2, to_gray=True)).float()
    state1[0][0] = state1[0][1]
    state1[0][1] = state1[0][2]
    state1[0][2] = tmp
    return state1


def prepare_initial_state(state, N=3):
    state_ = T.from_numpy(downscale_obs(state, to_gray=True)).float()
    tmp = state_.repeat((N, 1, 1))
    return tmp.unsqueeze(dim=0)
'''
"There is not an input shape anymore so we dont need it? For atari games "
" Error even when I have transposed the array "


def get_image(env):
    img = env.render(mode='rgb_array')
    print(img.shape)
    # convert an image from one colour space to another(from rgb to gray)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb_resized = cv2.resize(img_rgb, (84, 84), interpolation=cv2.INTER_CUBIC)
    # make all pixels black
    # img_rgb_resized[img_rgb_resized < 255] = 0
    # make pixel values between 0 and 1
    img_rgb_resized = img_rgb_resized / 255
    return img_rgb_resized


def get_img(env, input_shape):
    obs = env.render(mode='rgb_array')
    new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized_screen = resize(new_frame, (84, 84))
    # self.shape will be either 1 for grayscale or 3 for coloured image
    new_obs = np.array(resized_screen, dtype=np.uint8).reshape(input_shape)
    # make pixel values between 0 and 1
    new_obs = new_obs / 255.0
    return new_obs


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return T.from_numpy(screen)


def worker(name, input_shape, n_actions, global_agent, global_icm,
           optimizer, icm_optimizer, env_id, n_threads, icm=False):
    T.manual_seed(5)
    T_MAX = 20

    local_agent = ActorCritic(input_shape, n_actions)

    if icm:
        local_icm = ICM(input_shape, n_actions)
    else:
        local_icm = None
        intrinsic_reward = None

    memory = Memory()

    img_shape = [input_shape[1], input_shape[2], 1]
    env = make_env(env_id, shape=img_shape)

    episode, max_steps, t_steps, scores = 0, 1000, 0, []

    while episode < max_steps:
        obs = env.reset()
        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)
        while not done:
            state = T.tensor([obs], dtype=T.float)
            action, value, log_prob, hx = local_agent(state, hx)
            obs_, reward, done, info = env.step(action)
            memory.remember(obs, action, obs_, reward, value, log_prob)
            score += reward
            obs = obs_
            ep_steps += 1
            t_steps += 1
            if ep_steps % T_MAX == 0 or done:
                states, actions, new_states, rewards, values, log_probs = \
                    memory.sample_memory()
                if icm:
                    intrinsic_reward, L_I, L_F = \
                        local_icm.calc_loss(states, new_states, actions)

                loss = local_agent.calc_loss(obs, hx, done, rewards,
                                             values, log_probs,
                                             intrinsic_reward)
                optimizer.zero_grad()
                hx = hx.detach_()
                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()
                loss.backward()
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)
                for local_param, global_param in zip(
                        local_agent.parameters(),
                        global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(
                            local_icm.parameters(),
                            global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())

                memory.clear_memory()
        episode += 1
        # with global_idx.get_lock():
        #    global_idx.value += 1
        if name == '1':
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            # avg_score_5000 = np.mean(scores[max(0, episode - 5000): episode + 1])
            print('ICM episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'avg score (100)  {:.2f}'.format(
                episode, name, n_threads,
                t_steps / 1e6, score,
                avg_score))
    if name == '1':
        x = [z for z in range(episode)]
        plot_learning_curve(x, scores, 'ICM_hallway_final.png')