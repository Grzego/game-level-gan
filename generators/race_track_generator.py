import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import device, gumbel_noise


class GeneratorNetwork(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size
        self.output_size = 2
        self.internal_size = 256

        self.unpack_idea = nn.Sequential(
            nn.Linear(latent_size, self.internal_size),
            nn.ELU()
        )
        self.make_level = nn.LSTM(self.output_size, self.internal_size, num_layers=2)
        self.level_widths = nn.Sequential(
            nn.GroupNorm(self.internal_size // 64, self.internal_size),
            nn.Linear(self.internal_size, 1),
            nn.Sigmoid()
        )
        self.level_angles = nn.Sequential(
            nn.GroupNorm(self.internal_size // 64, self.internal_size),
            nn.Linear(self.internal_size, 1),
            nn.Tanh()
        )

    def forward(self, noise, num_segments):
        # noise = [batch_size, latent_size]
        idea = self.unpack_idea(noise)
        coords = noise.new_zeros((noise.size(0), self.output_size))

        idea = torch.stack((idea, torch.zeros_like(idea)), dim=0)

        level = []
        idea = (idea, idea)
        for _ in range(num_segments):
            coords, idea = self.make_level(coords.unsqueeze(0), idea)

            flatten = coords.squeeze(0)
            angles = self.level_angles(flatten)  # [batch_size, 1]
            widths = self.level_widths(flatten)  # [batch_size, 1]
            coords = torch.cat((angles, widths), dim=-1)
            level.append(coords)

        return torch.stack(level, dim=1)  # [batch_size, num_segments, internal_size]


class RaceTrackGenerator(object):
    """
    Generates levels for Race game.
    """

    def __init__(self, latent_size, lr=1e-4):
        self.latent_size = latent_size
        self.generator = GeneratorNetwork(latent_size).to(device)
        self.optimizer = optim.Adam(self.generator.parameters(), lr=lr)

    def generate(self, track_length, num_samples=1):
        """
        From random vector generate multiple samples of tracks with `track_length`.
        Track is a sequence of shape [num_samples, track_length, (arc, width)].
        """
        noise = torch.randn((num_samples, self.latent_size), device=device, requires_grad=True)
        return self.generator(noise, track_length)

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
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    @property
    def track_shape(self):
        return 2,
