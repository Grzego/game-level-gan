import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils import cudify, one_hot


class GeneratorNetwork(nn.Module):
    def __init__(self, latent_size, output_shape):
        super().__init__()
        self.height, self.width = output_shape

        self.unpack_idea = nn.Sequential(
            nn.Linear(latent_size, 128 * 4 * 4),
            nn.ELU()
        )
        self.make_level = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(64, 4, kernel_size=(3, 3), stride=2),
            # nn.Softmax(dim=-1)  # using gumbel sampling this is not needed
        )

    def forward(self, noise):
        idea = self.unpack_idea(noise)
        idea = idea.view(idea.size(0), 128, 4, 4)
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

        self.target_field_dist = Variable(cudify(torch.from_numpy(np.array([0.7, 0.2, 0.07, 0.03]))).float())
        self.mean_total_reward = 4.

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
        gumbel = cudify(torch.rand(num_samples, *level.shape))  # [num_samples, 4, height, width]
        gumbel.add_(1e-8).log_().neg_()
        gumbel.add_(1e-8).log_().neg_()
        gumbel = Variable(gumbel)
        _, argmax = torch.max(level + gumbel, dim=1)  # [num_samples, height, width]
        levels = one_hot(argmax, dim=1, num_classes=level.size(0))  # [num_samples, 4, height, width]

        # add players
        for x, y in [(0, 0), (-1, -1), (0, -1), (-1, 0)][:self.num_players]:
            levels[:, :, x, y] = 0

        # returns numpy array (gradient cannot flow anyway)
        levels = torch.cat((levels, self.players.expand(num_samples, *self.players.shape[1:])), dim=1)
        return levels.permute(0, 2, 3, 1).cpu().numpy()  # TODO: remove permute (change channel layout in Pacman)

    def generate(self, num_samples=1):
        """
        From random vector generate multiple samples of maps of `board_size`.
        Returned samples are wrapped in Variable to store grads.
        """
        noise = Variable(cudify(torch.randn(1, self.latent_size)))
        self.level = self.generator(noise).squeeze()
        new_levels = [Variable(cudify(torch.from_numpy(l)), requires_grad=True)
                      for l in self._sample(self.level, num_samples)]
        self.levels += new_levels
        return new_levels

    def backward(self, agents):
        """
        Use data stored in agents for gradients calculations.
        """
        loss = 0.
        for a in agents:
            policy, value, total_reward = a.generator_data()

            # aim of this loss is to promote levels that require agents
            # plans to be more complex
            loss += torch.mean(1. - torch.pow(policy[1:, :] - policy[:-1, :], 2.))

        loss.backward()

    def train(self):
        """
        Uses gradients stored in `self.levels` to train generator.
        """
        # TODO: implement (sum-up grads and update params) - maybe add auxiliary losses for reward placement
        # TODO: check if gradients are accumulated correctly

        # constraint total reward
        loss = 0.
        for l in self.levels:
            field_count = l.view(int(np.prod(self.board_size)), -1).sum(0)
            rewards = 0.5 * field_count[2] + field_count[3]
            loss += torch.pow(rewards - self.mean_total_reward, 2.)
        loss.backward()

        # sum-up gradients per level (from agent)
        grads = 0.
        for l in self.levels:
            grads += l.grad.data[:, :, :-self.num_players].permute(2, 0, 1)

        # force generator to follow given field distribution
        # level = Variable(self.level.data, requires_grad=True)
        # log_prob = level - level.exp().sum(0, keepdim=True).log()
        # field_prob = log_prob.view(-1, int(np.prod(self.board_size))).mean(1)
        # loss = F.kl_div(field_prob, self.target_field_dist)
        # loss.backward()

        # accumulate gradients from penalty
        # grads += level.grad.data

        # backward all gradients
        self.level.backward(grads)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.levels.clear()
