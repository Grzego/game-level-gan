import os
import random
import pyforms
from pyforms.controls import *
import torch
from torch import optim
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn
import cv2
import svgutils.transform as sg


from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest, one_hot, device


resume = os.path.join('learned')
num_players = 2
num_segments = 128
latent = 16


def interpolate_hypersphere(v1, v2, num_steps):
    v1_norm = torch.norm(v1)
    v2_norm = torch.norm(v2)
    v2_normalized = v2 * (v1_norm / v2_norm)

    vectors = []
    for step in range(num_steps):
        interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
        interpolated_norm = torch.norm(interpolated)
        interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
        vectors.append(interpolated_normalized)
    return torch.stack(vectors)


def main():
    global generator, discriminator, agents, game

    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=60. * 10., framerate=1. / 20., cars=cars)

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    # if resume:
    #     path = find_latest(resume, 'generator_[0-9]*.pt')
    #     print(f'Resuming generator from path "{path}"')
    #     generator.network.load_state_dict(torch.load(path, map_location=device))
    #     generator.deterministic = True

    # x, y = 4, 2
    # while True:
    #     tracks = generator.generate(track_length=num_segments, num_samples=16)
    #     game.reset(tracks)
    #
    #     imgs = game.prettier_tracks(x * y, size=1024)
    #     fig, ax = plt.subplots(y, x)
    #     for i in range(y):
    #         for j in range(x):
    #             ax[i, j].imshow(imgs[i * y + j])
    #             ax[i, j].axis('off')
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     plt.show()

    # while True:
    #     steps = 16
    #     n1, n2 = torch.randn(1, latent), torch.randn(1, latent)
    #     ins = n1 + (n2 - n1) * torch.linspace(0., 1., steps).view(-1, 1)
    #     # hns = interpolate_hypersphere(n1.view(-1), n2.view(-1), num_steps=steps)
    #
    #     tracks = generator.generate(track_length=num_segments, noise=ins)
    #     game.reset(tracks)
    #     imgs = game.prettier_tracks(steps, size=256)
    #     fig, ax = plt.subplots(1, steps)
    #     for i in range(steps):
    #         ax[i].imshow(imgs[i])
    #         ax[i].axis('off')
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     plt.show()

    # tracks = torch.zeros(4, num_segments, 2)
    # for i, t2 in enumerate([10, 30, 50, 70]):
    #     tracks[i, t2: t2 + 12, 0] = 1.
    #     tracks[i, t2 + 30: t2 + 42, 0] = -1.

    # tracks = generator.generate(64, 4)
    # tracks[:, :, 1] = 0.

    # tracks = torch.zeros(4, 64, 2)
    # for i in range(0, 64, 16):
    #     tracks[0, i: i + 10, 0] = 1.
    #
    # for i in range(0, 64, 32):
    #     tracks[3, i: i + 20, 0] = (i // 32) * (-2) + 1
    #
    # tracks[1:3, :, :] = generator.generate(64, 2)

    # tracks = torch.zeros(1, 256, 2)
    # tracks[0, 30:40, 0] = 1.
    # tracks[0, 60:70, 0] = 1.
    # tracks[0, 90:110, 0] = -1.
    # tracks[0, 150:160, 0] = -1.
    # tracks[0, 170:190, 0] = 1.
    # tracks[0, 205:215, 0] = 1.

    # tracks = torch.zeros(1, 128, 2)
    # tracks[0, 20:40, 0] = -1.
    # tracks[0, 50:65, 0] = 1.
    # tracks[0, 80:95, 0] = 1.
    # tracks[0, 120:, 0] = 1.

    # tracks = torch.tensor([[[0., 0.], [0.7, 0.], [-0.3, 0.]]])

    # size = 512
    # for i in range(1, 4):
    #     game.reset(tracks[:, :i, :])
    #     imgs = game.prettier_tracks_svg(top_n=1, size=size, pad=0.05)
    #     imgs[0].saveas('track_gen_{}.svg'.format(i))

    # game.reset(tracks)
    # imgs = game.prettier_tracks_svg(top_n=1, size=size, pad=0.05)
    # imgs[0].saveas('track.svg')

    # imgs = game.prettier_tracks(top_n=tracks.size(0), size=size, pad=0.05)
    # imgs = np.transpose(np.reshape(imgs, (2, 2, size, size, 4)), (0, 2, 1, 3, 4))
    # imgs = np.reshape(imgs, (2 * size, 2 * size, 4))
    # imgs = np.reshape(imgs, (size, size, 4))

    # cv2.imwrite('tracks.png', imgs)

    # create new SVG figure
    size = 500
    xx, yy = 1, 1
    gen_path = 'learned/generators'
    for gen_file in os.listdir(gen_path):
        if '7500' not in gen_file: continue
        gen_file = os.path.join('learned/generators', gen_file)
        print('Generating: {}'.format(gen_file))
        generator.network.load_state_dict(torch.load(gen_file, map_location=device))
        tracks = generator.generate(num_segments, xx * yy)
        game.reset(tracks)
        imgs = game.prettier_tracks_svg(top_n=xx * yy, size=size, pad=0.05)

        for i, img in enumerate(imgs):
            img.saveas('plots/gen_{}.svg'.format(i))

        fig = sg.SVGFigure()

        # load matpotlib-generated figures
        figs = [sg.fromfile('plots/gen_{}.svg'.format(i)) for i, _ in enumerate(imgs)]

        # get the plot objects
        plots = []
        for x in range(xx):
            for y in range(yy):
                plots += [figs[x * yy + y].getroot()]
                plots[-1].moveto(x * size, y * size)

        fig.append(plots)
        # save generated SVG files
        fig.save("gen_samples_{}.svg".format(gen_file.split('_')[-1].split('.')[0]))


if __name__ == '__main__':
    main()
