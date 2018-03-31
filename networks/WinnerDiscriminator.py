import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils import cudify


class DiscriminatorNetwork(nn.Module):
    def __init__(self, size, depth, num_players):
        super().__init__()
        self.size = size
        self.depth = depth
        self.num_players = num_players

        self.features = nn.Sequential(
            nn.Conv2d(depth, 64, kernel_size=(3, 3)),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),
            nn.ELU(),
            # nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1),
            # nn.ELU()
        )

        self.prediction = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ELU(),
            nn.Linear(256, num_players),
            nn.Softmax(dim=-1)
        )

    def feature_size(self):
        return int(np.prod(self.features(Variable(torch.ones(1, self.depth, *self.size))).shape))

    def forward(self, inputs):
        h = self.features(inputs)
        h = h.view(inputs.size(0), -1)
        return self.prediction(h)


class WinnerDiscriminator(object):
    def __init__(self, size, depth, num_players, lr=1e-4):
        self.network = cudify(DiscriminatorNetwork(size, depth, num_players))
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, boards):
        return self.network(boards)

    def loss(self, boards, winners):
        pred_winners = self.network(boards)
        return F.cross_entropy(pred_winners, winners)

    def train(self, boards, winners):
        loss = self.loss(boards, winners)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.data.cpu().numpy()[0]
