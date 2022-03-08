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
    img_rgb_resized = cv2.resize(img_rgb, (240, 160), interpolation=cv2.INTER_CUBIC)
    # make all pixels black
    # img_rgb_resized[img_rgb_resized < 255] = 0
    # make pixel values between 0 and 1
    img_rgb_resized = img_rgb_resized / 255
    return img_rgb_resized


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
        # just a string for printing debug information to the terminal && saving our plot
        algo = 'ICM'
    else:
        intrinsic_reward = T.zeros(1)
        algo = 'A3C'
    # each agent gets its own memory
    memory = Memory()
    # its own environment
    env = gym.make(env_id)
    # how many time steps we have, the episode , the score, the average score
    t_steps, max_eps, episode, scores, avg_score = 0, 1000, 0, [], 0
    # We have 1000 episodes/ time steps
    intr = []
    while episode < max_eps:
        state = env.reset()
        # make your hidden state for the actor critic a3c
        hx = T.zeros(1, 256)
        # we need a score, a terminal flag and the number of steps taken withing the episode
        # every 20 steps in an episode we want to execute the learning function
        score, done, ep_steps = 0, False, 0
        while not done:
            # state = T.tensor([obs], dtype=T.float)
            input_img = env.render(mode='rgb_array')
            # input_img = resize(input_img, (3, 240, 160)) # Resize for cartPole
            # input_img = input_img.transpose((0, 1, 2))
            # input_img = get_image(env)
            # print("input img worker render", input_img.shape)
            # input_img = resize(input_img, (3, 84, 84)) # Resize for b
            input_img = input_img.transpose((0, 1, 2))

            state = T.tensor([input_img], dtype=T.float)
            # print("state/input img shape in worker", state.shape)
            # input_img = get_screen(env)
            # print(input_img)
            # feed forward our state and our hidden state to the local agent to get the action we want to take,
            # value for that state, log_prob for that action
            action, value, log_prob, hx = local_agent(state, hx)
            # shape of input_img: (400,600,3) ... after resize:(240,160,3)
            # observation represents environments next state
            # take your action
            obs_, reward, done, info = env.step(action)
            # increment total steps, episode steps, increase your score
            t_steps += 1
            ep_steps += 1
            score += reward
            reward = 0  # turn off extrinsic rewards
            memory.remember(state, action, reward, obs_, value, log_prob)
            obs = obs_
            obs = obs.transpose((2, 0, 1))
            # print(obs.shape)
            # obs = T.tensor([obs])
            # print(obs.shape)
            # shape of obs: (4,)
            # LEARNING
            # every 20 steps or when the game is done
            if ep_steps % T_MAX == 0 or done:
                states, actions, rewards, new_states, values, log_probs = \
                    memory.sample_memory()
                # If we are doing icm then we want to calculate our loss according to icm
                if icm:
                    intrinsic_reward, L_I, L_F = \
                        local_icm.calc_loss(states, new_states, actions)
                # loss according to our a3c agent
                loss = local_agent.calc_loss(obs, hx, done, rewards, values,
                                             log_probs, intrinsic_reward)

                optimizer.zero_grad()
                hx = hx.detach_()
                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()

                """Update the weights of the network given a training sample. """
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
        # at every episode
        # for thread 1
        if name == '1':
            a = T.sum(intrinsic_reward)
            intr.append(a.detach().numpy())  # for plotting intrinsic reward
            # env.render()  # Render environment/ visualise
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print('{} episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'intrinsic_reward {:.2f} avg score (100) {:.1f}'.format(
                algo, episode, name, n_threads,
                t_steps / 1e6, score,
                T.sum(intrinsic_reward),
                avg_score))

        # end of one time step / episode
        episode += 1
    # At the end of the 1000 episodes
    if name == '1':
        # print(intr)
        x = [z for z in range(episode)]
        fname = algo + '_CartPole_no_rewards_.png'
        # fname1 = algo + '_CartPole_intrinsic_reward1'
        plot_learning_curve(x, scores, fname)
        # plot_intrinsic_reward(x, intr, fname1)
        # plot_learning_curve_with_shaded_error(x, scores, fname)
