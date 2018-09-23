import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import device, one_hot


class DiscriminatorNetwork(nn.Module):
    def __init__(self, num_players):
        super().__init__()

        self.input_size = 2
        self.hidden_size = 512
        # self.features = nn.LSTM(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        self.features = nn.ModuleList([
            nn.LSTMCell(self.input_size, self.hidden_size),
            nn.LSTMCell(self.hidden_size, self.hidden_size),
            nn.LSTMCell(self.hidden_size, self.hidden_size),
        ])
        self.prediction = nn.Sequential(
            # nn.GroupNorm(self.hidden_size // 64, self.hidden_size),
            # nn.Linear(self.hidden_size, self.hidden_size // 2),
            # nn.ELU(),
            nn.Linear(self.hidden_size, num_players)
            # nn.Softmax(dim=-1)
        )

    def flatten_parameters(self):
        self.features.flatten_parameters()

    def forward(self, tracks):
        # tracks = [batch_size, num_segments, 2]
        states = [(tracks.new_zeros(tracks.size(0), self.hidden_size),  # h
                   tracks.new_zeros(tracks.size(0), self.hidden_size))  # c
                  for _ in range(len(self.features))]
        for s in range(tracks.size(1)):
            h = tracks[:, s, :]
            for i, cell in enumerate(self.features):
                states[i] = cell(h, states[i])
                h = states[i][0]
        # h, _ = self.features(tracks)
        # h = F.elu(h[:, -1, :])
        h = F.elu(h)
        return self.prediction(h)


class ConvDiscriminatorNetwork(nn.Module):
    def __init__(self, num_players):
        super().__init__()
        base = 128
        self.features = nn.Sequential(
            nn.Conv1d(2, base, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv1d(base, 2 * base, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(2 * base, 4 * base, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(4 * base, 4 * base, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(4 * base, 8 * base, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(8 * base, 4 * base, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(4 * base, num_players, kernel_size=3)
        )

    def forward(self, tracks, min_segments=128):
        # tracks = [batch_size, num_segments, 2]
        h = tracks.permute(0, 2, 1)
        if h.size(-1) < min_segments:
            h = F.pad(h, (0, min_segments - h.size(-1)))
        h = self.features(h)
        return torch.sum(h, dim=-1)


class RaceWinnerDiscriminator(object):
    def __init__(self, num_players, lr=1e-4, asynchronous=False):
        self.num_players = num_players
        self.asynchronous = asynchronous
        self.lr = lr
        self.num_players = num_players
        # self.network = DiscriminatorNetwork(num_players + 1)  # +1 for invalid option
        self.network = ConvDiscriminatorNetwork(num_players + 1)  # +1 for invalid option
        self.optimizer = None
        self.stats = 1.

        if asynchronous:
            self.network.share_memory()
        self.network.to(device)
        # self.network.flatten_parameters()

        if not asynchronous:
            # self.stats = torch.ones(num_players + 1, device=device)
            # self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def async_optim(self, optimizer):
        # self.stats = torch.ones(self.num_players + 1, device=device)
        self.optimizer = optimizer

    def async_network(self, network):
        self.network = network

    def forward(self, tracks):
        # WARNING: recently changed to log_softmax
        return F.log_softmax(self.network(tracks), dim=-1)
        # return F.softmax(self.network(tracks), dim=-1)

    def loss(self, tracks, winners, trials=2):
        # WARNING: recently changed
        # wins = winners + 1
        # self.stats = 0.9 * self.stats + 0.1 * one_hot(wins, num_classes=self.num_players + 1).float().mean(0)
        # with torch.no_grad():
        #     self.stats = 0.99 * self.stats + 0.01 * winners.mean(0)

        log_probs = self.forward(tracks)
        # probs = self.forward(tracks)
        balance = 1.  # self.stats.add(0.01).reciprocal()
        loss = torch.mean(torch.mean(-winners * log_probs, dim=0) * balance)
        # loss = F.mse_loss(probs, winners)
        acc = log_probs.exp().mul(trials).round().div(trials).eq(winners).float().mean().item()
        # acc = log_probs.exp().sub(winners).abs().sum(1).mean().item()
        # acc = 1. - probs.sub(winners).abs().sum(1).mean()
        return loss, acc
        # return F.cross_entropy(logits_winners, wins), wins.eq(pred_winners).float().mean()
        # return F.cross_entropy(logits_winners, wins, 1. / (self.stats + 0.01)), wins.eq(pred_winners).float().mean()

    def train(self, tracks, winners, trials=2):
        loss, acc = self.loss(tracks, winners, trials=trials)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), acc
