import os
import cv2
import svgwrite
import torch
from torch import optim
from torch.nn import functional as F
import numpy as np

from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest, one_hot, device


import matplotlib.pyplot as plt
import seaborn

seaborn.set()
seaborn.set_style("whitegrid", {'axes.grid': False})

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
trials = 50  # 20
# trials = 1
latent = 16
observation_size = 18
generator_batch_size = 64
generator_train_steps = 1


def slerp(low, high, steps):
    nsteps = torch.linspace(0., 1., steps)[:, None].to(device)
    nlow = low / torch.norm(low)
    nhigh = high / torch.norm(high)
    omega = torch.acos(torch.dot(nlow, nhigh))
    so = torch.sin(omega)
    return torch.sin((1. - nsteps) * omega) / so * low + torch.sin(nsteps * omega) / so * high


def slerp2(low, high, step):
    nlow = low / torch.norm(low)
    nhigh = high / torch.norm(high)
    omega = torch.acos(torch.dot(nlow, nhigh))
    so = torch.sin(omega)
    return torch.sin((1. - step) * omega) / so * low + torch.sin(step * omega) / so * high


def main():
    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=4., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=40., framerate=1. / 20., cars=cars)
    gen_game = Race(timeout=40., framerate=1. / 20., cars=cars)

    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    # load agents if resuming
    if learned_agents:
        for i, a in enumerate(agents):
            path = find_latest(learned_agents, 'agent_{}_*.pt'.format(i))
            print(f'Resuming agent {i} from path "{path}"')
            a.network.load_state_dict(torch.load(path))
            a.old_network.load_state_dict(torch.load(path))
            # a.network.cuda()
            # a.old_network.cuda()

            torch.save(a.network.state_dict(), 'agent_{}.pt'.format(i))

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if learned_generator:
        path = find_latest(learned_generator, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))
        generator.network.cuda()

    noise = torch.load('noise/noise_best.pt')

    # 0 - top, 1 - bot
    corners = [[noise[-1], noise[2]], [noise[0], noise[1]]]
    # corners = [[noise[-1], noise[-1]], [noise[2], noise[2]]]

    gx, gy = 16, 16
    # vecs = torch.zeros((gy, gx, 16), device=device)
    for q in range(10):
        print(f'Generating {q}')
        vecs = torch.randn(gy, gx, 16, device=device)

        # vecs[:, 0, :] = slerp(corners[0][1], corners[1][0], gy)

        # for yy, y in enumerate(torch.linspace(0., 1., gy)):
        #     for xx, x in enumerate(torch.linspace(0., 1., gx)):
        #         # vecs[yy, xx, :] = (corners[0][0] * x + corners[0][1] * (1 - x)) * y + \
        #         #                   (corners[1][0] * x + corners[1][1] * (1 - x)) * (1 - y)
        #         vecs[yy, xx, :] = slerp2(slerp2(corners[0][0], corners[0][1], x),
        #                                  slerp2(corners[1][0], corners[1][1], x), y)

        vecs = vecs.view(gx * gy, -1)
        tracks = generator.generate(max_segments, t=100., noise=vecs)

        size = 512
        game.reset(tracks)
        img = game.prettier_tracks(top_n=gx * gy, size=size)
        img = np.reshape(img, (gy, gx, size, size, 4))
        img = np.transpose(img, (0, 2, 1, 3, 4))
        img = np.reshape(img, (gy * size, gx * size, 4))

        plt.imshow(img)
        plt.show()

        # generated boards
        with torch.no_grad():
            generated_boards = tracks  # generator.generate(max_segments, eval_size, t=10.)

            generated_winners = 0.
            for t in range(trials):
                states, any_valid = game.reset(generated_boards)
                print(f'\r[{t + 1:2d}/{trials:2d}] Generated boards eval...')
                step = 0
                while any_valid and not game.finished():
                    print(f'\r[{step:4d}]', end='')
                    step += 1
                    actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
                    states, rewards = game.step(actions)
                    for a, r in zip(agents, rewards):
                        a.observe(r)
                print()

                for a in agents:
                    a.reset()

                generated_winners += one_hot(game.winners() + 1, num_classes=num_players + 1).float()
            generated_winners /= trials

            print(generated_winners.float().mean(0))
            print(generated_winners.float().std(0))

            plt.subplot(1, 2, 1)
            plt.hist(generated_winners[:, 1].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 1')
            plt.legend()
            plt.xlim(0., 1.)
            plt.subplot(1, 2, 2)
            plt.hist(generated_winners[:, 2].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 2')
            plt.xlim(0., 1.)
            # plt.show()
            plt.legend()
            plt.savefig('generated.png')
            plt.close()

            blue = torch.tensor([127, 176, 255], dtype=torch.float, device=device) / 255.
            red = torch.tensor([255, 124, 124], dtype=torch.float, device=device) / 255.
            green = torch.tensor([181, 255, 147], dtype=torch.float, device=device) / 255.

            winners = generated_winners.view(gy, gx, -1)
            fairness = winners.clone()  # .permute(1, 0, 2).contiguous()
            fairness *= 2.
            fairness -= 1.
            fairness.clamp_(0., 1.)

            # fairness[fairness > 0] = 1. - fairness[fairness > 0]

            fairmap = fairness.new_zeros(*fairness.shape[:2], 4, dtype=torch.float)
            fairmap[:, :, 3] = 1.
            fairmap[:, :, :3] += (fairness[:, :, 1:2] > 0).float() * ((1. - fairness[:, :, 1:2]) + red * fairness[:, :, 1:2])
            fairmap[:, :, :3] += (fairness[:, :, 2:3] > 0).float() * ((1. - fairness[:, :, 2:3]) + blue * fairness[:, :, 2:3])
            fairmap[:, :, :3] += (fairness[:, :, 0:1] > 0).float() * ((1. - fairness[:, :, 0:1]) + green * fairness[:, :, 0:1])
            fairmap = (fairmap * 255.).byte()
            fairmap[(fairmap.int().sum(-1, keepdim=True) == 255).repeat(1, 1, 4)] = 255

            fairmap = fairmap.view(gy, 1, gx, 1, -1).repeat(1, size, 1, size, 1).view(gy * size, gx * size, -1)

        img = fairmap.cpu().numpy() * (img[:, :, 3:4] < 10) + img * (img[:, :, 3:4] >= 10)
        plt.imshow(fairmap.cpu().numpy())
        plt.show()
        plt.imshow(img)
        # plt.close()
        plt.show()

        cv2.imwrite(f'tracks-with-fairness-{q}.png', img)


if __name__ == '__main__':
    main()
