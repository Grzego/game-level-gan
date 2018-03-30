import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

from utils import cudify, one_hot, gumbel_noise


class GeneratorNetwork(nn.Module):
    def __init__(self, latent_size, output_shape):
        super().__init__()
        self.height, self.width = output_shape

        self.unpack_idea = nn.Sequential(
            nn.Linear(latent_size, 256 * 4 * 4),
            nn.ELU()
        )
        self.make_level = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 4, kernel_size=(3, 3), stride=2),
            # nn.Softmax(dim=-1)  # using gumbel sampling this is not needed
        )

    def forward(self, noise):
        idea = self.unpack_idea(noise)
        idea = idea.view(idea.size(0), 256, 4, 4)
        level = self.make_level(idea)

        crop_h = (level.size(2) - self.height) // 2
        crop_w = (level.size(3) - self.width) // 2
        return level[:, :, crop_h: crop_h + self.height, crop_w: crop_w + self.width]


class SimplePacmanGenerator(object):
    """
    Simple generator for Pacman game.

    It generates board with shape = (H, W, 4)
    H - height
    W - width
    4 - levels of map, [0 - empty, 1 - wall, 2 - small reward, 3 - large reward]
    """

    def __init__(self, latent_size, board_size, num_players, lr=1e-4):
        """
        `num_players` in range [2, 4] -- each players is put in corner.
        """
        self.latent_size = latent_size
        self.board_size = board_size
        self.num_players = num_players
        self.levels = []
        self.level = None

        self.target_field_dist = Variable(cudify(torch.from_numpy(np.array([0.4, 0.5, 0.07, 0.03]))).float())
        self.level_score_cap = 4.

        self.generator = cudify(GeneratorNetwork(latent_size, board_size))
        self.optimizer = optim.Adam(self.generator.parameters(), lr=lr)

        players = torch.zeros(1, num_players, *board_size)
        for i, (x, y) in enumerate([(0, 0), (-1, -1), (0, -1), (-1, 0)][:self.num_players]):
            players[0, i, x, y] = 1
        self.players = cudify(players)

    def _sample(self, level, num_samples):
        """
        Level represents probabilities over each field.
        This function should sample this distributions
        and as a result return `num_samples` levels.
        """
        gumbel = gumbel_noise(level.shape)  # [num_samples, 4, height, width]
        _, argmax = torch.max(level + gumbel, dim=1)  # [num_samples, height, width]
        levels = one_hot(argmax, dim=1, num_classes=level.size(1))  # [num_samples, 4, height, width]

        # add players
        for x, y in [(0, 0), (-1, -1), (0, -1), (-1, 0)][:self.num_players]:
            levels[:, :, x, y] = 0

        levels = torch.cat((levels, self.players.expand(num_samples, *self.players.shape[1:])), dim=1)
        return levels.permute(0, 2, 3, 1)  # TODO: remove permute (change channel layout in Pacman)

    def generate(self, num_samples=1):
        """
        From random vector generate multiple samples of maps of `board_size`.
        Returned samples are wrapped in Variable to store grads.
        """
        noise = Variable(cudify(torch.randn(num_samples, self.latent_size)))
        self.level = self.generator(noise)
        self.levels = Variable(self._sample(self.level.data, num_samples), requires_grad=True)

        # TODO: check second approach - generate few samples and then their variations and sum grads over variations
        # new_levels = [Variable(cudify(torch.from_numpy(l)), requires_grad=True)
        #               for l in self._sample(self.level, num_samples)]
        # self.levels += new_levels
        return self.levels

    def backward(self, agents):
        """
        Use data stored in agents for gradients calculations.
        """
        # TODO: add loss logging (maybe use tensorboardX package)

        loss = 0.
        for a in agents:
            policy, value, rewards = a.generator_data()

            # aim of this loss is to promote levels that require agents
            # plans to be more complex
            loss += 0.1 * torch.mean(1. - torch.pow(policy[:, 1:, :] - policy[:, :-1, :], 2.))

            # maybe maximize player score?
            # loss += -0.01 * torch.mean(value)

        loss.backward()

    def train(self):
        """
        Uses gradients stored in `self.levels` to train generator.
        """
        # TODO: implement (sum-up grads and update params) - maybe add auxiliary losses for reward placement
        # TODO: check if gradients are accumulated correctly

        # constraint rewards to be around `self.level_score_cap`
        # rewards = 0.5 * self.levels[:, 2, :, :] + self.levels[:, 3, :, :]
        # rewards = rewards.view(self.levels.size(0), -1).sum(dim=-1)
        # loss = 10. * torch.mean(torch.pow(rewards - self.level_score_cap, 2.))
        # loss.backward()

        # penalize symatric boards
        if self.board_size[0] == self.board_size[1]:
            level_part = self.levels[:, :, :, :4]
            loss = torch.pow(level_part - level_part.permute(0, 2, 1, 3), 2.)
            loss *= Variable(1. - cudify(torch.eye(self.board_size[0])[None, ..., None]))  # diag should not count
            loss = 0.1 * torch.mean(loss)
            loss.backward()

        grads = 0.
        grads += self.levels.grad.data[:, :, :, :-self.num_players].permute(0, 3, 1, 2)

        # force generator to follow given field distribution
        level = Variable(self.level.data, requires_grad=True)
        log_prob = level - level.exp().sum(0, keepdim=True).log()
        field_prob = log_prob.view(-1, 4, int(np.prod(self.board_size))).mean(dim=2).mean(dim=0)
        loss = F.kl_div(field_prob, self.target_field_dist)
        loss.backward()

        # accumulate gradients from penalty
        grads += level.grad.data

        # backward all gradients
        self.level.backward(grads)
        self.optimizer.step()
        self.optimizer.zero_grad()
