import os
import cv2
import torch
from torch import optim
from torch.nn import functional as F
import numpy as np

from game import Race, RaceCar
from generators import RaceTrackGenerator
from utils import find_next_run_dir, find_latest, one_hot, device

import matplotlib.pyplot as plt
# import seaborn

# seaborn.set()

# learned_agents = None  # os.path.join('..', 'experiments', '013', 'run-4')
# learned_agents = os.path.join('..', 'experiments', '015', 'run-2')
learned_agents = os.path.join('learned')
learned_discriminator = None  # os.path.join('experiments', 'run-5')
# learned_discriminator = os.path.join('experiments', 'run-6')
learned_generator = os.path.join('learned')  # os.path.join('experiments', 'run-5')
# learned_generator = os.path.join('experiments', 'run-3')
resume_segments = 128
# resume_segments = 128
num_players = 2
batch_size = 32
# batch_size = 32 - 4 * 6
max_segments = 128
num_proc = 1
trials = 6
# trials = 1
latent = 16
observation_size = 18
generator_batch_size = 64
generator_train_steps = 1


def main():
    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=60. * 10., framerate=1. / 20., cars=cars)
    gen_game = Race(timeout=60. * 10., framerate=1. / 20., cars=cars)

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if learned_generator:
        path = find_latest(learned_generator, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))
        generator.network.cuda()

    # noise = torch.randn(4, 16).to(device)
    # tracks = generator.generate(track_length=128, num_samples=4, t=1., noise=noise)
    # game.reset(tracks)
    #
    # plots = game.prettier_tracks(top_n=4, size=512)
    # plt.imshow(np.reshape(plots, (-1, *plots.shape[2:])))
    # plt.show()

    # TODO:
    # - script that you can draw some shape
    # - it will break it down into 128 segments (as racetrack)
    # - run optimization to find latent code for it

    # img = np.ones((512, 511, 3), np.uint8) * 255
    # cv2.namedWindow('Track')
    #
    # drawing = False
    # px, py = None, None
    #
    # lines = []
    #
    # def draw_lines(event, x, y, flags, params):
    #     nonlocal drawing, px, py
    #
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         drawing = True
    #     elif event == cv2.EVENT_MOUSEMOVE and drawing:
    #         cv2.line(img, (px, py), (x, y), (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)
    #         lines.append(((x, y), (px, py)))
    #     elif event == cv2.EVENT_LBUTTONUP:
    #         drawing = False
    #
    #     px, py = x, y
    #
    # cv2.setMouseCallback('Track', draw_lines)
    #
    # while True:
    #     cv2.imshow('Track', img)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    #
    # print('lines:', lines)

    tracks = torch.zeros(8, max_segments, 2, device=device)

    # zig-zag
    t0 = 0  # random.randint(0, 15)
    for i in range(t0, max_segments, 16):
        tracks[0, i: i + 16, 0] = 2. * ((i // 16) % 2) - 1.

    # one sharp turn
    t1 = 55  # random.randint(0, 100)
    tracks[1, t1: t1 + 20, 0] = 1.

    # S track
    t2 = 30  # random.randint(0, 70)
    tracks[2, t2: t2 + 12, 0] = 1.
    tracks[2, t2 + 30: t2 + 42, 0] = -1.

    # large U turn
    # t3 = [random.randint(0, 5) for _ in range(7)]
    t3 = [1, 2, 0, 4, 0, 1, 4]
    offs = [0, 10, 20, 40, 50, 70, 100]
    for o, t in zip(offs, t3):
        tracks[3, t + o: t + o + 5, 0] = 1.

    # small bumpy turn
    t4 = 43  # random.randint(0, 90)
    tracks[4, t4: t4 + 6, 0] = 1.
    tracks[4, t4 + 6: t4 + 18, 0] = -1.
    tracks[4, t4 + 18: t4 + 24, 0] = 1.

    # immediate turn
    t5 = 50  # random.randint(30, 60)
    tracks[5, :20, 0] = 1.
    tracks[5, -t5: -t5 + 20, 0] = -1.

    # clockwise O track
    for i in range(0, max_segments, 3):
        tracks[6, i, 0] = 1

    # widdershins O track
    for i in range(0, max_segments, 3):
        tracks[7, i, 0] = -1

    game.reset(tracks)
    img = game.prettier_tracks(top_n=8)

    noise = torch.zeros(8, 16, requires_grad=True, device=device)
    optimizer = optim.LBFGS([noise], lr=0.2, max_iter=50, history_size=1000)

    max_steps = 1000
    for step in range(1, 1 + max_steps):
        gen = None

        def closure():
            nonlocal gen
            optimizer.zero_grad()
            gen = generator.generate(max_segments, t=1. + 4. * (step / max_steps), noise=noise)
            # gen = generator.generate(max_segments, t=10., noise=noise)
            loss = F.mse_loss(tracks, gen)
            loss.backward()
            print('\r[{:5d}] loss = {}'.format(step, loss.item()), end='')
            return loss
        optimizer.step(closure)

        if step % 100 == 0:
            gen_game.reset(gen.detach())
            gen_img = gen_game.prettier_tracks(top_n=8)

            all_img = np.stack((img, gen_img), axis=0)
            all_img = np.transpose(all_img, (0, 2, 1, 3, 4))
            all_img = np.reshape(all_img, (2 * 1024, 8 * 1024, 4))
            plt.imshow(all_img)
            plt.show()

            torch.save(noise, 'noise/noise_{}.pt'.format(step))


if __name__ == '__main__':
    main()
