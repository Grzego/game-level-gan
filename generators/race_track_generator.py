import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import device, one_hot


class GeneratorNetwork(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size
        # self.noise_size = 64
        self.input_size = 2
        self.output_size = 2
        self.rnn_size = 512
        self.code_size = 2048
        self.mixtures = 5

        self.unpack_idea = nn.Sequential(
            nn.Linear(latent_size, self.code_size),
            nn.Tanh()
        )
        # nn.LSTM(self.code_size, self.rnn_size, num_layers=2)
        self.make_level = nn.ModuleList([
            nn.LSTMCell(self.code_size, self.rnn_size),
            nn.LSTMCell(self.rnn_size, self.rnn_size)
        ])
        self.attention = nn.Sequential(
            nn.Linear(self.rnn_size, self.code_size),
            nn.Sigmoid()
        )
        # self.level_widths = nn.Sequential(
        #     nn.Linear(self.rnn_size, 1),
        #     nn.Sigmoid()
        # )
        self.angles_means = nn.Linear(self.rnn_size, self.mixtures)
        self.angles_vars = nn.Linear(self.rnn_size, self.mixtures)
        self.angles_mix = nn.Sequential(
            nn.Linear(self.rnn_size, self.mixtures),
            # nn.Softmax(dim=-1)
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
        states = [(latent.new_zeros(latent.size(0), self.rnn_size),
                   latent.new_zeros(latent.size(0), self.rnn_size))
                  for _ in range(len(self.make_level))]
        for s in range(num_segments):
            flatten = torch.mul(idea, attention)
            for i, cell in enumerate(self.make_level):
                states[i] = cell(flatten, states[i])
                flatten = states[i][0]

            # angles = self.level_angles(flatten)
            # widths = self.level_widths(flatten)
            attention = self.attention(flatten)

            rho = F.softmax(self.angles_mix(flatten), dim=-1)
            mu = torch.sum(rho * self.angles_means(flatten), dim=-1, keepdim=True)
            sigma = torch.sum(rho * torch.exp(self.angles_vars(flatten) - t), dim=-1, keepdim=True)

            samples = torch.distributions.Normal(loc=mu, scale=sigma).rsample()
            angles = F.tanh(samples)

            level.append(torch.cat((angles, torch.zeros_like(angles)), dim=-1))  # constant width for now

        return torch.stack(level, dim=1)  # [batch_size, num_segments, internal_size]


class GeneratorNetworkDiscrete(nn.Module):
    def __init__(self, latent_size, discrete_size):
        super().__init__()

        self.latent_size = latent_size
        # self.noise_size = 64
        self.input_size = 2
        self.output_size = 2
        self.discrete_size = discrete_size
        self.rnn_size = 512
        self.code_size = 512

        self.unpack_idea = nn.Sequential(
            nn.Linear(latent_size + self.input_size, self.code_size),
            nn.Tanh()
        )
        self.make_level = nn.ModuleList([
            nn.LSTMCell(self.code_size, self.rnn_size),
            nn.LSTMCell(self.rnn_size, self.rnn_size)
        ])
        self.angle = nn.Linear(self.rnn_size, self.discrete_size)
        self.space = torch.linspace(-1., 1., self.discrete_size).view(1, -1).to(device)

    def forward(self, latent, num_segments, t=0.):
        # latent = [batch_size, latent_size]
        # coord_noise = latent.new_empty((latent.size(0), num_segments, self.output_size))
        # coord_noise[:, :, 0].uniform_(-1., 1.)
        # coord_noise[:, :, 1].uniform_(0., 1.)

        level = [latent.new_zeros((latent.size(0), self.input_size))]
        states = [(latent.new_zeros(latent.size(0), self.rnn_size),
                   latent.new_zeros(latent.size(0), self.rnn_size))
                  for _ in range(len(self.make_level))]
        for s in range(num_segments):
            flatten = self.unpack_idea(torch.cat((latent, level[-1]), dim=-1))
            for i, cell in enumerate(self.make_level):
                states[i] = cell(flatten, states[i])
                flatten = states[i][0]

            angles = torch.sum(F.gumbel_softmax(self.angle(flatten), hard=True) * self.space, dim=-1, keepdim=True)
            # angles = torch.sum(one_hot(torch.argmax(self.angle(flatten), dim=-1), num_classes=self.discrete_size).float() * self.space,
            #                    dim=-1, keepdim=True)
            level.append(torch.cat((angles, torch.zeros_like(angles)), dim=-1))  # constant width for now

        return torch.stack(level[1:], dim=1)  # [batch_size, num_segments, internal_size]


class RaceTrackGenerator(object):
    """
    Generates levels for Race game.
    """

    def __init__(self, latent_size, lr=1e-4, asynchronous=False):
        self.latent_size = latent_size
        # self.network = GeneratorNetworkDiscrete(latent_size, discrete_size=64)
        self.network = GeneratorNetwork(latent_size)
        self.optimizer = None

        if asynchronous:
            self.network.share_memory()
        self.network.to(device)
        # self.network.flatten_parameters()

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
        # reverse_mask = torch.ones_like(pred_winners)
        # reverse_mask[:, 0].neg_()

        # shift_mask = torch.zeros_like(pred_winners)
        # shift_mask[:, 0] = 1.

        # prob_wins = pred_winners * reverse_mask + shift_mask
        # loss = -torch.mean(torch.log(prob_wins + 1e-8))

        wanted = torch.full_like(pred_winners, 1. / (pred_winners.size(1) - 1.))
        wanted[:, 0] = 0.

        # pred_winners should be log_prob
        loss = -torch.mean(wanted * pred_winners)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @property
    def track_shape(self):
        return 2,
