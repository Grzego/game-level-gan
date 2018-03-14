import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

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
        noise = Variable(cudify(torch.rand(num_samples, *level.shape)))  # [num_samples, 4, height, width]
        gumbel = -torch.log(-torch.log(noise))
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
        level = self.generator(noise).squeeze()
        new_levels = [Variable(cudify(torch.from_numpy(l)), requires_grad=True)
                      for l in self._sample(level, num_samples)]
        self.levels += new_levels
        return new_levels

    def backward(self, agents):
        """
        Use data stored in agents for gradients calculations.
        """
        # TODO: implement
        for a in agents:
            _ = a.generator_data()
        ...

    def train(self):
        """
        Uses gradients stored in `self.levels` to train generator.
        """
        # TODO: implement (sum-up grads and update params)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.levels.clear()
