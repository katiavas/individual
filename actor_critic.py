import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
'''
Convolutional features are just that, they're convolutions, maybe max-pooled convolutions, but they aren't flat.
We need to flatten them, like we need to flatten an image before passing it through a regular layer'''

# This is for Breakout-v5 resize 84x84 image
# 2 convolutional layers to extract features from the images
class Encoder(nn.Module):

    def __init__(self, input_dims, feature_dim=64):
        super(Encoder, self).__init__()
        # kernel size: 1x1 kernel/window that rolls over data to find features
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        # 3 colour channels/rgb values
        # Using the last 3 frames gives our models access to velocity information (i.e. how fast and which direction things are moving) rather than just positional information.
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # input is 3 images, 32 output channels, 3x3 kernel / window
        shape = self.conv_output(input_dims)
        #  determine the actual shape of the flattened output after the first convolutional layers.
        self.fc1 = nn.Linear(shape, feature_dim)  # shape after resize 240x160
        # self.fc1 = nn.Linear(225792, feature_dim)  # shape after resize for 84x84

    def conv_output(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        return int(np.prod(dims.size()))

    def forward(self, img):
        enc = self.conv1(img)
        enc = self.conv2(enc)
        # Flattens input by reshaping it into a 1-d tensor. If start_dim are passed, only dimensions starting with start_dim are flattened
        # enc_flatten = enc.flatten(start_dim=1)
        # enc_flatten = T.flatten(enc, start_dim=1)
        # print('put this shape into the fc1 layer: ', enc_flatten.size())
        enc_flatten = enc.view(enc.size()[0], -1)
        features = self.fc1(enc_flatten)
        # print("features", features.shape)
        return features


# AC3 is the actual on-policy algorithm to generate the policy π
# Used for environments with discrete action spaces --> AC3
# This class is going to inherit from the nn module
class ActorCritic(nn.Module):
    # The __init__ method lets the class initialize the object’s attributes
    # tau is the constant lamda from the paper
    def __init__(self, input_dim, n_actions, gamma=0.99, tau=0.98, feature_dim=64):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.encoder = Encoder(input_dim, feature_dim)
        # Our network will need an input layer which will take an input and translate that into 256
        # self.input = nn.Linear(*input_dims, 256)
        self.input = nn.Linear(feature_dim, 256)
        # A dense layer
        self.dense = nn.Linear(256, 256)
        # Lstm type layer receives the reward
        self.gru = nn.GRUCell(256, 256)
        # Policy
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)
        device = T.device('cpu')
        self.to(device)

    # It will take a state/image and a hidden state for our GRU as an input
#    def forward(self, state, hx):
    def forward(self, img, hx):
        # print("actor critic forward image", img)
        state = self.encoder(img)
        # print("Forward model state/img shape", state.shape)
        # x = F.relu(self.input(state))
        # x = F.relu(self.dense(x))
        hx = self.gru(state, hx)
        # Pass hidden state into our pi and v layer to get our logs for our policy(pi) and out value function
        pi = self.pi(hx)
        v = self.v(hx)
        # Choose action function/ Get the actual probability distribution
        probs = T.softmax(pi, dim=1)  # soft max activation on the first dimension
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # return predicted action, value, log probability and hidden state
        return action.numpy()[0], v, log_prob, hx

    # Functions to handle the calculation of the loss
    # https://arxiv.org/pdf/1602.01783.pdf
    def calc_R(self, done, rewards, values):  # done/terminal flag, set of rewards, set of values--> stored in a list of tensors
        # we want to convert this list of tensors to a single tensor and squeeze it because we dont want T time stepsby1
        values = T.cat(values).squeeze()
        # A3C must get triggered every T timestep or everytime an episode ends / we could have a batch of states or a single state
        # if we have batch of states then the length of values.size is one
        if len(values.size()) == 1:  # batch of states
            R = values[-1] * (1-int(done))
        elif len(values.size()) == 0:  # single state
            R = values*(1-int(done))
        # Calculate the returns at each time step of R sequence
        batch_return = []
        # Iterate backwards in our rewards
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()  # reverse it
        batch_return = T.tensor(batch_return,
                                dtype=T.float).reshape(values.size())
        return batch_return

    def calc_loss(self, new_states, hx, done,
                  rewards, values, log_probs, r_i_t=None):
        # if we are supplying an intrinsic reward them we want to add the reward from ICM
        if r_i_t is not None:
            # convert r_i_t to a numpy array because r_i_t is a tensor while rewards is a list of floating point values
            rewards += r_i_t.detach().numpy()
        returns = self.calc_R(done, rewards, values)
        # calculate generalised advantage
        # We need a value function for the state one step after our horizon
        # get the first element because other elements that the forward function returns are not the value function
        # (we want the element v )
        # next_v = T.zeros(1, 1) if done else self.forward(new_states, hx)[1]
        next_v = T.zeros(1, 1) if done else self.forward(T.tensor([new_states], dtype=T.float), hx)[1]

        values.append(next_v.detach())
        values = T.cat(values).squeeze()  # concatenate -> cat
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)
        #                   state of time at t+1  state of time at t
        delta_t = rewards + self.gamma*values[1:] - values[:-1]
        n_steps = len(delta_t)
        # generalised advantage estimate : https://arxiv.org/pdf/1506.02438.pdf
        # There is gonna be an advantage for each time step in the sequence
        # So gae is gonna be a batch of states, T in length
        # So we have an advantage for each time step, which is proportional to a sum of all the rewards that follow
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k*delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)
        # Calculate losses
        actor_loss = -(log_probs*gae).sum()
        entropy_loss = (-log_probs*T.exp(log_probs)).sum()
        # [T] vs ()
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)  # mean squared error
        total_loss = actor_loss + critic_loss - 0.01*entropy_loss

        return total_loss



'''

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, tau=1.0):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.tau = tau

        self.conv1 = nn.Conv2d(input_dims[0], 32, (1, 1))
        self.conv2 = nn.Conv2d(32, 32, (1, 1))
        self.conv3 = nn.Conv2d(32, 32, (1, 1))
        self.conv4 = nn.Conv2d(32, 32, (1, 1))

        conv_shape = self.calc_conv_output(input_dims)

        self.gru = nn.GRUCell(conv_shape, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def calc_conv_output(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        dims = self.conv4(dims)
        return int(np.prod(dims.size()))

    def forward(self, state, hx):
        conv = F.elu(self.conv1(state))
        conv = F.elu(self.conv2(conv))
        conv = F.elu(self.conv3(conv))
        conv = F.elu(self.conv4(conv))

        conv_state = conv.view((conv.size()[0], -1))

        hx = self.gru(conv_state, hx)

        pi = self.pi(hx)
        v = self.v(hx)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], v, log_prob, hx

    def calc_R(self, done, rewards, values):
        values = T.cat(values).squeeze()

        if len(values.size()) == 1:  # batch of states
            R = values[-1]*(1-int(done))
        elif len(values.size()) == 0:  # single state
            R = values*(1-int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return,
                                dtype=T.float).reshape(values.size())
        return batch_return

    def calc_cost(self, new_state, hx, done,
                  rewards, values, log_probs, intrinsic_reward=None):

        if intrinsic_reward is not None:
            rewards += intrinsic_reward.detach().numpy()

        returns = self.calc_R(done, rewards, values)

        next_v = T.zeros(1, 1) if done else self.forward(T.tensor(
                                        [new_state], dtype=T.float), hx)[1]
        values.append(next_v.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)

        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k * delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)

        actor_loss = -(log_probs * gae).sum()
        # if single then values is rank 1 and returns rank 0
        # want to have same shape to avoid a warning
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)

        entropy_loss = (-log_probs * T.exp(log_probs)).sum()

        total_loss = actor_loss + critic_loss - 0.01 * entropy_loss
        return total_loss
'''
