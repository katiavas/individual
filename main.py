import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv
import torch
from memory import Memory


os.environ['OMP_NUM_THREADS'] = '1'


if __name__ == '__main__':
    # Set seed
    torch.manual_seed(5)
    # mp.set_start_method('forkserver')
    mp.set_start_method('spawn')
    memory = Memory()
    env_id = 'ALE/Breakout-v5'
    # env_id = 'CartPole-v1'
    n_threads = 2
    n_actions = 4
    # n_actions = 2
    # print(input_shape)
    input_shape = [3, 84, 84]
    env = ParallelEnv(env_id=env_id, n_threads=n_threads,
                      n_actions=n_actions, input_shape=input_shape, icm=True)
                      
                      
# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
'''the state-space of the Cart-Pole has four dimensions of continuous values 
and the action-space has one dimension of two discrete values'''
