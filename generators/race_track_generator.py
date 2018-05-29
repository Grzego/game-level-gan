import torch
from torch import nn
from torch import optim

from utils import device


class GeneratorNetwork(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size
        # self.noise_size = 64
        self.input_size = 2
        self.output_size = 2
        self.rnn_size = 512
        self.code_size = 2048

        self.unpack_idea = nn.Sequential(
            nn.Linear(latent_size, self.code_size),
            nn.Tanh()
        )
        self.make_level = nn.LSTM(self.code_size, self.rnn_size, num_layers=2)
        self.attention = nn.Sequential(
            nn.Linear(self.rnn_size, self.code_size),
            nn.Sigmoid()
        )
        self.level_widths = nn.Sequential(
            nn.Linear(self.rnn_size, 1),
            nn.Sigmoid()
        )
        self.level_angles = nn.Sequential(
            nn.Linear(self.rnn_size, 1),
            nn.Tanh()
        )

    def flatten_parameters(self):
        self.make_level.flatten_parameters()

    def forward(self, latent, num_segments, t=0.):
        # latent = [batch_size, latent_size]
        # coord_noise = latent.new_empty((latent.size(0), num_segments, self.output_size))
        # coord_noise[:, :, 0].uniform_(-1., 1.)
        # coord_noise[:, :, 1].uniform_(0., 1.)
        attention = latent.new_ones(latent.size(0), self.code_size)
        idea = self.unpack_idea(latent)

        level = []
        state = None
        for s in range(num_segments):
            segment, state = self.make_level(torch.mul(idea, attention).unsqueeze(0), state)

            flatten = segment.squeeze(0)
            angles = self.level_angles(flatten)
            widths = self.level_widths(flatten)
            attention = self.attention(flatten)

            level.append(torch.cat((angles, widths), dim=-1))

        return torch.stack(level, dim=1)  # [batch_size, num_segments, internal_size]


class RaceTrackGenerator(object):
    """
    Generates levels for Race game.
    """

    def __init__(self, latent_size, lr=1e-4, asynchronous=False):
        self.latent_size = latent_size
        self.network = GeneratorNetwork(latent_size).to(device)
        self.optimizer = None

        if not asynchronous:
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def async_optim(self, optimizer):
        self.optimizer = optimizer

    def generate(self, track_length, num_samples=1, t=0.):
        """
        From random vector generate multiple samples of tracks with `track_length`.
        Track is a sequence of shape [num_samples, track_length, (arc, width)].
        """
        noise = torch.randn((num_samples, self.latent_size), device=device)
        return self.network(noise, track_length, t)

    def train(self, pred_winners):
        """
        Generator wants all players to have equal chance of winning.
        Last dim means whether board was invalid, this probability should be 0.
        """
        reverse_mask = torch.ones_like(pred_winners)
        reverse_mask[:, 0].neg_()

        shift_mask = torch.zeros_like(pred_winners)
        shift_mask[:, 0] = 1.

        prob_wins = pred_winners * reverse_mask + shift_mask
        loss = -torch.mean(torch.log(prob_wins + 1e-8))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @property
    def track_shape(self):
        return 2,
