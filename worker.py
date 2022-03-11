import gym
import numpy as np
import torch as T
from actor_critic import ActorCritic
from icm import ICM
from memory import Memory
from utils import plot_learning_curve
from utils import plot_intrinsic_reward
from utils import plot_intrinsic_reward_avg
from utils import plot_learning_curve_with_shaded_error
from utils import plot_learning_curve1
import torch as T
import cv2
import csv
# from utils import plot_learning_curve_with_shaded_error
import matplotlib.pyplot as plt
from wrapper import make_env

from skimage.transform import resize



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
    # env = gym.make(env_id)

    img_shape = [input_shape[1], input_shape[2], 1]
    env = make_env(env_id, shape=img_shape)

    scores2 = []
    episode, max_steps, t_steps, scores = 0, 5000, 0, []
    intr = []
    while episode < max_steps:
        obs = env.reset()
        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)
        while not done:
            # input_img = env.render(mode='rgb_array')
            # input_img = resize(input_img, (1, 84, 84)) # Resize for breakout
            # input_img = input_img.transpose((0, 1, 2))
            # state = T.tensor([input_img], dtype=T.float)
            # print(state.shape)
            state = T.tensor([obs], dtype=T.float)
            action, value, log_prob, hx = local_agent(state, hx)
            obs_, reward, done, info = env.step(action)
            memory.remember(obs, action, obs_, reward, value, log_prob)
            score += reward
            obs = obs_
            # obs = resize(obs, (1, 84, 84))
            # obs = obs.transpose((0, 1, 2))
            # obs = T.tensor([obs], dtype=T.float)
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
            a = T.sum(intrinsic_reward)
            intr.append(a.detach().numpy())  # for plotting intrinsic reward
            scores.append(score)
            if episode <= 1000:
                scores2.append(score)
            avg_score = np.mean(scores[-100:])
            # avg_score_5000 = np.mean(scores[max(0, episode - 5000): episode + 1])
            print('ICM episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'avg score (100)(5000) {:.2f}'.format(
                episode, name, n_threads,
                t_steps / 1e6, score,
                avg_score))
    if name == '1':
        x = [z for z in range(episode)]
        np.savetxt("GFG.csv",
                   scores,
                   delimiter=", ",
                   fmt='% s')
        plot_learning_curve(x, scores, 'ICM_Final2.png')
        # plot_intrinsic_reward_avg(x, intr, 'ICM_intr_avg1.png')
        # plot_learning_curve_with_shaded_error(x, scores, 'Learning_curve_shaded_error_ICM.png')
        plot_learning_curve1(x, scores, scores2, 'Plot.plt')


"""Hi, I have just finished implementing the encoders for the icm and have output some graphs, plus some graphs for cartpole implementation. Since there is no available meeting for the next two weeks and we havent met for the last 2-3 weeks, I was wondering if you have time to do a quick meeting next week? After outputing those graphs, I dont know how to proceed and to be fair I am not entirely sure what I am actually looking for, eventhough I have done progress and have followed the papaer's implementation."""