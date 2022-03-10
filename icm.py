import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
# self.inverse = nn.Linear(6*2, 256)
# self.dense1 = nn.Linear(6+1, 256)
# self.new_state = nn.Linear(256, *input_dims)
# Breakout --> n_actions = 4 , input_shape/input_dims = rgb

class Encoder(nn.Module):

    def __init__(self, input_dims, feature_dim=64):
        super(Encoder, self).__init__()
        # kernel size: 1x1 kernel/window that rolls over data to find features
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        # 3 colour channels/rgb values
        # Using the last 3 frames gives our models access to velocity information (i.e. how fast and which direction things are moving) rather than just positional information.
        self.conv1 = nn.Conv2d(1, 32, (1, 1))  # input is 3 images, 32 output channels, 1x1 kernel / window
        self.conv2 = nn.Conv2d(32, 32, (1, 1))
        shape = self.conv_output(input_dims)
        #  determine the actual shape of the flattened output after the first convolutional layers.
        # self.fc1 = nn.Linear(7680000, feature_dim) # shape for CartPole-v0
        # self.fc1 = nn.Linear(1075200, feature_dim) # shape for Breakout-v0
        self.fc1 = nn.Linear(shape, feature_dim)  # shape after resize 240x160
        # self.fc1 = nn.Linear(225792, feature_dim)  # shape after resize for 84x84

    def conv_output(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        return int(np.prod(dims.size()))

    def forward(self, img):
        enc = F.elu(self.conv1(img))
        enc = self.conv2(enc)
        # Flattens input by reshaping it into a 1-d tensor. If start_dim are passed, only dimensions starting with start_dim are flattened
        # enc_flatten = enc.flatten(start_dim=1)
        # enc_flatten = T.flatten(enc, start_dim=1)
        # print('put this shape into the fc1 layer: ', enc_flatten.size())
        # Bc fc1 needs a linear input
        enc_flatten = enc.view(enc.size()[0], -1)
        # print(enc_flatten)
        features = self.fc1(enc_flatten)
        # print("features", features.shape)
        return features


'''In the inverse model you want to predict the action the agent took to cause this state to transition from time t to t+1
So you are comparing an integer vs an actual label/ the actual action the agent took
Multi-class classification problem
This is a cross entropy loss between the predicted action and the actual action the agent took'''
"The loss for the forward model is the mse between the predicted state at time t+1 and the actua state at time t+1  "
"So we have two losses : one that comes from the inverse model and one that comes from the forward model "


# Cartpole n_actions = 2, input_dims = 4
class ICM(nn.Module):
    def __init__(self, input_dims, n_actions=4, alpha=1, beta=0.2, feature_dim=64):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.encoder = Encoder(input_dims, feature_dim=64)
        # print("Features", self.encoder)
        self.inverse = nn.Linear(feature_dim * 2, 256)
        self.pi_logits = nn.Linear(256, n_actions)

        self.dense1 = nn.Linear(feature_dim + 1, 256)
        self.new_state = nn.Linear(256, feature_dim)

        device = T.device('cpu')
        self.to(device)

    # Forward model takes the action and the current state and predicts the next state
    # def forward(self, state, new_state, action):
    def forward(self, obs, new_obs, action):
        """ We have to concatenate a state and action and pass it through the inverse layer """
        "and activate it with an elu activation--> exponential linear"
        obs = T.Tensor(obs)
        # print(obs.shape, "obs")
        # obs = obs.view(obs.size()[0], -1).to(T.float)
        # obs = new_obs.view(new_obs.size()[0], -1).to(T.float)
        # print("obs", obs.shape)
        state = self.encoder(obs)
        with T.no_grad():
            new_state = self.encoder(new_obs)

        state = state.view(state.size()[0], -1).to(T.float)
        new_state = new_state.view(new_state.size()[0], -1).to(T.float)
        # print(new_state.shape, "new")
        # Create inverse layer
        inverse = F.elu(self.inverse(T.cat([state, new_state], dim=1)))
        pi_logits = self.pi_logits(inverse)

        # Forward model
        # from [T] to [T,1]
        action = action.reshape((action.size()[0], 1))
        # Activate the forward input and get a new state on the other end
        forward_input = T.cat([state, action], dim=1)
        dense = F.elu(self.dense1(forward_input))
        state_ = self.new_state(dense)
        # print(state_.shape, "s")

        return new_state, pi_logits, state_

    def calc_loss(self, state, new_state, action):
        state = T.tensor(state, dtype=T.float)
        action = T.tensor(action, dtype=T.float)
        new_state = T.tensor(new_state, dtype=T.float)
        # feed/pass state, new_state , action through our network
        phi_new, pi_logits, state_ = self.forward(state, new_state, action)
        "Our inverse loss is a cross entropy loss because this will generally have more than two actions"
        inverse_loss = nn.CrossEntropyLoss()
        L_I = (1 - self.beta) * inverse_loss(pi_logits, action.to(T.long))
        "Forward loss is mse between predicted new state and actual new state"
        forward_loss = nn.MSELoss()
        L_F = self.beta * forward_loss(state_, phi_new)
        # dim=1 for mean(dim=1) is very important. If you take that out it will take the mean across all dimensions
        # and you just get a single number, which is not useful
        # because the curiosity reward is associated with each state, so you have to take the mean across that first
        # dimension which is the number of states
        intrinsic_reward = self.alpha * ((state_ - phi_new).pow(2)).mean(dim=1)
        return intrinsic_reward, L_I, L_F
