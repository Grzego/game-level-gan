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
        self.code_size = 512
        self.mixtures = 5

        self.unpack_idea = nn.Sequential(
            nn.Linear(latent_size, self.code_size),
            nn.Tanh()
        )
        # nn.LSTM(self.code_size, self.rnn_size, num_layers=2)
        self.make_level = nn.ModuleList([
            nn.LSTMCell(self.code_size + self.input_size, self.rnn_size),
            nn.LSTMCell(self.rnn_size, self.rnn_size)
        ])
        # self.attention = nn.Sequential(
        #     nn.Linear(self.rnn_size, self.code_size),
        #     nn.Sigmoid()
        # )
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
        idea = self.unpack_idea(latent)

        level = [latent.new_zeros(latent.size(0), self.input_size)]
        states = [(latent.new_zeros(latent.size(0), self.rnn_size),
                   latent.new_zeros(latent.size(0), self.rnn_size))
                  for _ in range(len(self.make_level))]
        for s in range(num_segments):
            flatten = torch.cat((idea, level[-1]), dim=-1)
            for i, cell in enumerate(self.make_level):
                states[i] = cell(flatten, states[i])
                flatten = states[i][0]

            rho = F.softmax(self.angles_mix(flatten), dim=-1)
            mu = torch.sum(rho * self.angles_means(flatten), dim=-1, keepdim=True)
            sigma = torch.sum(rho * torch.exp(self.angles_vars(flatten) - t), dim=-1, keepdim=True)

            samples = torch.distributions.Normal(loc=mu, scale=sigma).rsample()
            angles = F.tanh(samples)

            level.append(torch.cat((angles, torch.zeros_like(angles)), dim=-1))  # constant width for now

        return torch.stack(level[1:], dim=1), None  # [batch_size, num_segments, internal_size]


class GeneratorNetworkAttention(nn.Module):
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

        return torch.stack(level, dim=1), None  # [batch_size, num_segments, internal_size]


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

        entropy_loss = 0.

        for s in range(num_segments):
            flatten = self.unpack_idea(torch.cat((latent, level[-1]), dim=-1))
            for i, cell in enumerate(self.make_level):
                states[i] = cell(flatten, states[i])
                flatten = states[i][0]

            angle_logits = self.angle(flatten)

            angle_log_prob = F.log_softmax(angle_logits, dim=-1)
            entropy_loss += torch.sum(angle_log_prob.exp() * angle_log_prob, dim=-1)

            angles = torch.sum(F.gumbel_softmax(angle_logits, hard=True) * self.space, dim=-1, keepdim=True)
            # angles = torch.sum(one_hot(torch.argmax(self.angle(flatten), dim=-1),
            #                            num_classes=self.discrete_size).float() * self.space,
            #                    dim=-1, keepdim=True)
            level.append(torch.cat((angles, torch.zeros_like(angles)), dim=-1))  # constant width for now

        return torch.stack(level[1:], dim=1), entropy_loss.mean() / num_segments  # [batch_size, num_segments, internal_size]


class GeneratorNetworkConvDiscrete(nn.Module):
    def __init__(self, latent_size, discrete_size, max_segments=128):
        super().__init__()

        self.latent_size = latent_size
        # self.noise_size = 64
        self.input_size = 2
        self.output_size = 2
        self.discrete_size = discrete_size
        self.code_size = 512

        self.unpack_idea = nn.Sequential(
            nn.Linear(latent_size, 8 * self.code_size),
            nn.Tanh()
        )
        self.make_level = nn.Sequential(
            nn.ConvTranspose1d(self.code_size, self.code_size, kernel_size=5, stride=2, padding=2),  # 15
            nn.ELU(),
            nn.ConvTranspose1d(self.code_size, self.code_size // 2, kernel_size=3, stride=2, padding=1),  # 29
            nn.ELU(),
            nn.ConvTranspose1d(self.code_size // 2, self.code_size // 4, kernel_size=3, stride=2, padding=1),  # 57
            nn.ELU(),
            nn.ConvTranspose1d(self.code_size // 4, self.code_size // 4, kernel_size=3, stride=2, padding=1),  # 113
            nn.ELU(),
            nn.ConvTranspose1d(self.code_size // 4, self.code_size // 8, kernel_size=3, stride=2, padding=1),  # 225
            nn.ELU(),
            nn.ConvTranspose1d(self.code_size // 8, self.discrete_size, kernel_size=3, stride=1, padding=1),  # 225
        )
        self.space = torch.linspace(-1., 1., self.discrete_size).view(1, -1).to(device)

    def forward(self, latent, num_segments, t=0.):
        # latent = [batch_size, latent_size]

        batch_size = latent.size(0)

        idea = self.unpack_idea(latent)  # [batch_size, 8 * code_size]
        h = idea.view(-1, self.code_size, 8)
        h = self.make_level(h)  # [batch_size, 64, 225]

        h = h[:, :, :num_segments].permute(0, 2, 1)  # [batch_size, num_segments, 64]

        # log_prob = F.log_softmax(h, dim=-1).view(-1, self.discrete_size)
        # entropy = torch.mean(torch.sum(log_prob.exp() * log_prob, dim=-1))

        if t > 0.:
            h = F.softmax(h.contiguous().view(-1, self.discrete_size) * t, dim=-1) * self.space
        else:
            h = F.gumbel_softmax(h.contiguous().view(-1, self.discrete_size), hard=True) * self.space
        h = torch.sum(h, dim=-1).view(batch_size, -1, 1)

        # we want to maximize this (more diversity between generated tracks
        # this sould be in range [0., 1.]
        aux_loss = 0.5 * torch.mean(torch.abs(h[None, :, :, 0] - h[:, None, :, 0]), dim=-1)  # [batch_size, batch_size]
        aux_loss = torch.sum(aux_loss) / (batch_size * (batch_size - 1) / 8.)  # scale loss properly

        return torch.cat((h, torch.zeros_like(h)), dim=-1), -aux_loss  # entropy


class RaceTrackGenerator(object):
    """
    Generates levels for Race game.
    """

    def __init__(self, latent_size, lr=1e-4, betas=(0.9, 0.999), asynchronous=False):
        self.latent_size = latent_size
        # self.network = GeneratorNetworkDiscrete(latent_size, discrete_size=64)
        # self.network = GeneratorNetworkConvDiscrete(latent_size, discrete_size=64)
        self.network = GeneratorNetworkConvDiscrete(latent_size, discrete_size=9)
        # self.network = GeneratorNetwork(latent_size)
        # self.network = GeneratorNetworkAttention(latent_size)
        self.optimizer = None
        self.auxiliary_loss = None

        if asynchronous:
            self.network.share_memory()
        self.network.to(device)
        # self.network.flatten_parameters()

        if not asynchronous:
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr, betas=betas)

    def async_optim(self, optimizer):
        self.optimizer = optimizer

    def generate(self, track_length, num_samples=1, t=0., noise=None):
        """
        From random vector generate multiple samples of tracks with `track_length`.
        Track is a sequence of shape [num_samples, track_length, (arc, width)].
        """
        if noise is None:
            noise = torch.randn((num_samples, self.latent_size), device=device)
        boards, self.auxiliary_loss = self.network(noise, track_length, t)
        return boards

    def train(self, pred_winners, beta=0.001):
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

        if self.auxiliary_loss:
            loss += beta * self.auxiliary_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), beta * self.auxiliary_loss.item() if self.auxiliary_loss is not None else None

    @property
    def track_shape(self):
        return 2,

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'latent': self.latent_size
        }, path)

    def load(self, path):
        data = torch.load(path)
        self.latent_size = data['latent']
        self.network.load_state_dict(data['network'])
        if data['optimizer'] is not None:
            self.optimizer.load_state_dict(data['optimizer'])

    @staticmethod
    def from_file(path):
        data = torch.load(path)
        model = RaceTrackGenerator(data['latent'])
        model.load(path)
        return model
