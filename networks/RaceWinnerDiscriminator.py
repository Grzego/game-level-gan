import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import device


class DiscriminatorNetwork(nn.Module):
    def __init__(self, num_players):
        super().__init__()

        self.input_size = 2
        self.hidden_size = 512
        self.features = nn.LSTM(self.input_size, self.hidden_size, num_layers=2, batch_first=True)
        self.prediction = nn.Sequential(
            nn.GroupNorm(self.hidden_size // 64, self.hidden_size),
            nn.Linear(self.hidden_size, num_players),
            nn.Softmax(dim=-1)
        )

    def forward(self, tracks):
        # tracks = [batch_size, num_segments, 2]
        h, _ = self.features(tracks)
        h = F.elu(h[:, -1, :])
        return self.prediction(h)


class RaceWinnerDiscriminator(object):
    def __init__(self, num_players, lr=1e-4):
        self.network = DiscriminatorNetwork(num_players).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, tracks):
        return self.network(tracks)

    def loss(self, tracks, winners):
        prob_winners = self.network(tracks)
        pred_winners = torch.argmax(prob_winners, dim=-1)
        return F.cross_entropy(prob_winners, winners), winners.eq(pred_winners).float().mean()

    def train(self, tracks, winners):
        loss, acc = self.loss(tracks, winners)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), acc.item()
