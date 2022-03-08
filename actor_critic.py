import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

'''Convolutional features are just that, they're convolutions, maybe max-pooled convolutions, but they aren't flat. 
We need to flatten them, like we need to flatten an image before passing it through a regular layer'''


# 2 convolutional layers to extract features from the images
class Encoder(nn.Module):

    def __init__(self, feature_dim=64):
        super(Encoder, self).__init__()
        # kernel size: 1x1 kernel/window that rolls over data to find features
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        # 3 colour channels/rgb values
        # Using the last 3 frames gives our models access to velocity information (i.e. how fast and which direction things are moving) rather than just positional information.
        self.conv1 = nn.Conv2d(3, 32, (1, 1))  # input is 3 images, 32 output channels, 1x1 kernel / window
        self.conv2 = nn.Conv2d(32, 32, (1, 1))
        # print(self.conv1)
        #  determine the actual shape of the flattened output after the first convolutional layers.
        # self.fc1 = nn.Linear(7680000, feature_dim) # shape for CartPole-v0
        # self.fc1 = nn.Linear(1075200, feature_dim) # shape for Breakout-v0
        # self.fc1 = nn.Linear(1228800, feature_dim)  # shape after resize 240x160
        self.fc1 = nn.Linear(225792, feature_dim)  # shape after resize for 84x84

    def forward(self, img):
        # print("expected input", img.shape)
        enc = self.conv1(img)
        # print(img.shape)
        enc = self.conv2(enc)
        # Flattens input by reshaping it into a 1-d tensor. If start_dim are passed, only dimensions starting with start_dim are flattened
        enc_flatten = enc.flatten(start_dim=1)
        # enc_flatten = T.flatten(enc, start_dim=1)
        # print('put this shape into the fc1 layer: ', enc_flatten.size())
        features = self.fc1(enc_flatten)
        # print("features", features.shape)
        return features

    '''  def __init__(self, feature_dim=64):
        super(Encoder, self).__init__()
        # kernel size: 1x1 kernel/window that rolls over data to find features
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        # 3 colour channels/rgb values
        self.conv1 = nn.Conv2d(3, 32, (1, 1))  # input is 3 images, 32 output channels, 1x1 kernel / window
        self.conv2 = nn.Conv2d(32, 32, (1, 1))
        #  determine the actual shape of the flattened output after the first convolutional layers.
        img = T.rand(50, 150).view(-1, 3, 50, 50)
        self._to_linear = None
        self.convs(img)
        self.fc1 = nn.Linear(self._to_linear, feature_dim)

    # Number of Linear input connections depends on output of conv2d layers
    # and therefore the input image size, so compute it.
    def convs(self, img):
        print("img before relu", img.shape)
        img = self.conv1(img)
        img = self.conv2(img)
        print("img shape after filtering in convs", img.shape)
        # To flatten the image we need to just grab the dimensions and multiply them
        if self._to_linear is None:
            # self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0]
            # self._to_linear = self._to_linear.detach().numpy()
            self._to_linear = img[0].shape[0]*img[0].shape[1]*img[0].shape[2]
            # print(self._to_linear)
        # print(img.shape)
        return img

    def forward(self, img):
        print("img shape in forward encoder before", img.shape)
        img = self.convs(img)
        img = img.view(1, self._to_linear)  # .view is reshape ... this flattens X before
        print("img shape in forward encoder after reshaping", img.shape)
        img = self.fc1(img)
        return F.softmax(img, dim=1)'''


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
        self.encoder = Encoder(feature_dim)
        # Our network will need an input layer which will take an input and translate that into 256
        # self.input = nn.Linear(*input_dims, 256)
        self.input = nn.Linear(feature_dim, 256)
        # A dense layer
        self.dense = nn.Linear(256, 256)
        # Lstm type layer receives the reward
        self.gru = nn.GRUCell(256, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)
        device = T.device('cpu')
        self.to(device)

    # It will take a state/image and a hidden state for our GRU as an input
#    def forward(self, state, hx):
    def forward(self, img, hx):
        state = self.encoder(img)
        # print("Forward model state/img shape", state.shape)
        x = F.relu(self.input(state))
        x = F.relu(self.dense(x))
        hx = self.gru(x, hx)
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
        next_v = T.zeros(1, 1) if done else self.forward(new_states, hx)[1]

        values.append(next_v.detach())
        values = T.cat(values).squeeze()  # concatenate -> cat
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)
        #                   state of time at t+1  state of time at t
        delta_t = rewards + self.gamma*values[1:] - values[:-1]
        n_steps = len(delta_t)
        '''generalised advantage estimate : https://arxiv.org/pdf/1506.02438.pdf'''
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
