import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

from utils import device


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
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=1),
            nn.ELU()
        )

        self.prediction = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ELU(),
            nn.Linear(512, num_players),
            nn.Softmax(dim=-1)
        )

    def feature_size(self):
        return int(np.prod(self.features(torch.ones(1, self.depth, *self.size)).shape))

    def forward(self, inputs):
        h = self.features(inputs)
        h = h.view(inputs.size(0), -1)
        return self.prediction(h)


class WinnerDiscriminator(object):
    def __init__(self, size, depth, num_players, lr=1e-4):
        self.network = DiscriminatorNetwork(size, depth, num_players).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, boards):
        return self.network(boards)

    def loss(self, boards, winners):
        prob_winners = self.network(boards)
        _, pred_winners = torch.max(prob_winners, dim=-1)
        return F.cross_entropy(prob_winners, winners), winners.long().eq(pred_winners.long()).float().mean()

    def train(self, boards, winners):
        loss, acc = self.loss(boards, winners)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.data.cpu().numpy()[0], acc.data.cpu().numpy()[0]
