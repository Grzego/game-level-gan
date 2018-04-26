import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import device


class DiscriminatorNetwork(nn.Module):
    def __init__(self, track_length, num_players):
        super().__init__()
        self.features = nn.Sequential(
        )

        self.prediction = nn.Sequential(
        )

    def feature_size(self):
        # return int(np.prod(self.features(Variable(torch.ones(1, self.depth, *self.size))).shape))
        pass

    def forward(self, inputs):
        h = self.features(inputs)
        h = h.view(inputs.size(0), -1)
        return self.prediction(h)


class RaceWinnerDiscriminator(object):
    def __init__(self, track_length, num_players, lr=1e-4):
        self.network = DiscriminatorNetwork(track_length, num_players).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, tracks):
        return self.network(tracks)

    def loss(self, tracks, winners):
        prob_winners = self.network(tracks)
        _, pred_winners = torch.max(prob_winners, dim=-1)
        return F.cross_entropy(prob_winners, winners), winners.long().eq(pred_winners.long()).float().mean()

    def train(self, tracks, winners):
        loss, acc = self.loss(tracks, winners)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), acc.item()
