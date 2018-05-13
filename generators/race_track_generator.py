import torch
from torch import nn
from torch import optim

from utils import device, Bipolar


class GeneratorNetwork(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size
        self.noise_size = 64
        self.input_size = 2
        self.output_size = 2
        self.internal_size = 512

        self.unpack_idea = nn.Sequential(
            # TODO: add some attention over latent code?
            nn.Linear(latent_size + self.input_size + self.noise_size, self.internal_size),
            Bipolar(nn.ELU())
        )
        self.make_level = nn.LSTM(self.internal_size, self.internal_size, num_layers=3)
        self.level_widths = nn.Sequential(
            # nn.GroupNorm(self.internal_size // 64, self.internal_size),
            nn.Linear(self.internal_size, 1),
            nn.Sigmoid()
        )
        self.level_angles = nn.Sequential(
            # nn.GroupNorm(self.internal_size // 64, self.internal_size),
            nn.Linear(self.internal_size, 1),
            nn.Tanh()
        )

    def forward(self, latent, num_segments):
        # latent = [batch_size, latent_size]
        level = []
        state = None
        coords = latent.new_zeros((latent.size(0), self.output_size))
        for _ in range(num_segments):
            noise = latent.new_empty((latent.size(0), self.noise_size))
            noise.normal_()

            idea = self.unpack_idea(torch.cat((coords, latent, noise), dim=-1))
            coords, state = self.make_level(idea.unsqueeze(0), state)

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
        noise = torch.randn((num_samples, self.latent_size), device=device)
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

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @property
    def track_shape(self):
        return 2,
