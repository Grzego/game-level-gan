import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class LSTMPolicy(nn.Module):
    def __init__(self, input_size, in_channels, num_actions, gumbel=True):
        super(LSTMPolicy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 128, kernel_size=(3, 3))

        conv_features = self._conv_pass(input_size, in_channels)
        self.lstm = nn.LSTM(conv_features, 256)
        self.policy = nn.Linear(self.lstm.hidden_size, num_actions)
        self.value = nn.Linear(self.lstm.hidden_size, 1)

        self.gumbel = gumbel
        self.state = None
        self.reset_state()

    def _conv_pass(self, input_size, in_channels):
        h = Variable(torch.zeros(1, in_channels, *input_size))
        h = self.conv1(h)
        h = self.conv2(h)
        return int(np.prod(h.shape[1:]))

    def reset_state(self):
        self.state = None

    def forward(self, inputs):
        h = F.elu(self.conv1(inputs))
        h = F.elu(self.conv2(h))
        h = h.view(1, -1, self.lstm.input_size)
        _, self.state = self.lstm(h, self.state)
        h, _ = self.state
        policy = self.policy(h)
        if not self.gumbel:
            policy = F.softmax(policy, dim=-1)
        value = self.value(h)
        return policy, value

