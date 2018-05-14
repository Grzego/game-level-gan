import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import device
from utils.pytorch_utils import one_hot


class DiscriminatorNetwork(nn.Module):
    def __init__(self, num_players):
        super().__init__()

        self.input_size = 2
        self.hidden_size = 512
        self.features = nn.LSTM(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        self.prediction = nn.Sequential(
            # nn.GroupNorm(self.hidden_size // 64, self.hidden_size),
            nn.Linear(self.hidden_size, num_players),
            nn.Softmax(dim=-1)
        )

    def flatten_parameters(self):
        self.features.flatten_parameters()

    def forward(self, tracks):
        # tracks = [batch_size, num_segments, 2]
        h, _ = self.features(tracks)
        h = F.elu(h[:, -1, :])
        return self.prediction(h)


class RaceWinnerDiscriminator(object):
    def __init__(self, num_players, lr=1e-4, asynchronous=False):
        self.num_players = num_players
        self.asynchronous = asynchronous
        self.lr = lr
        self.network = DiscriminatorNetwork(num_players + 1)  # +1 for invalid option
        self.optimizer = None
        # self.stats = torch.ones(num_players + 1, device=device)

        if asynchronous:
            self.network.share_memory()
        self.network.to(device)

        if not asynchronous:
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def async_optim(self, optimizer):
        self.optimizer = optimizer

    def async_network(self, network):
        self.network = network

    def forward(self, tracks):
        return self.network(tracks)

    def loss(self, tracks, winners):
        wins = winners + 1
        # self.stats = 0.9 * self.stats + 0.1 * one_hot(wins, num_classes=self.num_players + 1).float().mean(0)

        prob_winners = self.network(tracks)
        pred_winners = torch.argmax(prob_winners, dim=-1)
        return F.cross_entropy(prob_winners, wins), wins.eq(pred_winners).float().mean()
        # return F.cross_entropy(prob_winners, wins, 1. / (self.stats + 0.01)), wins.eq(pred_winners).float().mean()

    def train(self, tracks, winners):
        loss, acc = self.loss(tracks, winners)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), acc.item()
