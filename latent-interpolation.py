import os
import cv2
import torch
import numpy as np

from games import race_game as game
from games import RaceConfig
from generators import RaceTrackGenerator
from utils import find_latest, one_hot, device

import matplotlib.pyplot as plt
import seaborn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--generator', default=None, required=True, type=str)
parser.add_argument('--steps', default=10, type=int)
args = parser.parse_args()

seaborn.set()
seaborn.set_style("whitegrid", {'axes.grid': False})


def slerp(low, high, steps):
    nsteps = torch.linspace(0., 1., steps)[None, :, None].to(low.device)
    nlow = low / torch.norm(low)
    nhigh = high / torch.norm(high)
    omega = torch.acos(torch.sum(nlow * nhigh, dim=-1))[:, None, None]
    so = torch.sin(omega)
    return torch.sin((1. - nsteps) * omega) / so * low[:, None, :] + torch.sin(nsteps * omega) / so * high[:, None, :]


def main():
    # load generator
    path = find_latest(args.generator, 'generator_[0-9]*.pt')
    print(f'Resuming generator from path "{path}"')
    generator = RaceTrackGenerator.from_file(path)
    latent = generator.latent_size

    rows = 6
    v1 = torch.randn(rows, latent) * 0.01
    v2 = torch.randn(rows, latent) * 0.01

    points = slerp(v1, v2, args.steps).view(-1, latent).to(device)
    tracks = generator.generate(RaceConfig.max_segments, t=100., noise=points)

    size = 512
    game.reset(tracks)
    img = game.prettier_tracks(top_n=args.steps * rows, size=size)
    img = np.reshape(img, (rows, args.steps, size, size, 4))
    img = np.transpose(img, (0, 2, 1, 3, 4))
    img = np.reshape(img, (size * rows, args.steps * size, 4))

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
