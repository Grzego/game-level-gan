import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import device, gumbel_noise


class GeneratorNetwork(nn.Module):
    def __init__(self, latent_size, output_shape):
        super().__init__()

        self.unpack_idea = nn.Sequential(
        )
        self.make_level = nn.Sequential(
            # nn.Softmax(dim=-1)  # using gumbel sampling this is not needed
        )

    def forward(self, noise):
        pass


class RaceTrackGenerator(object):
    """

    """

    def __init__(self, latent_size, lr=1e-4):
        pass

    def generate(self, track_length, num_samples=1):
        """
        From random vector generate multiple samples of tracks with `track_length`.
        Track is a sequence of shape [num_samples, track_length, (arc, width)].
        """
        pass

    def train(self, pred_winners):
        """
        """
        pass

    @property
    def track_shape(self):
        return 2,
