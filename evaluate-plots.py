import os
import random
import torch
from torch import optim
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter

import glob
import matplotlib as mpl
mpl.use('Agg')
import seaborn
seaborn.set()
mpl.rcParams['font.family'] = 'Neris'
import matplotlib.pyplot as plt
import json

from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest, one_hot, device


# resume = os.path.join('..', 'experiments', '013', 'run-4')
resume_agents = os.path.join('learned')
resume_generator = os.path.join('experiments', 'run-0')
# resume = os.path.join('experiments', 'run-11')
num_players = 2
num_segments = 128
latent = 16
eval_size = 1000
trials = 20


def smooth(data, window=30):
    return np.array([data[i: i + window].mean() for i in range(len(data))])


def main():
    global generator, discriminator, agents, game

    data = np.array(json.load(open('experiments/run_run-0_summary-tag-discriminator_loss.json', 'r')))
    plt.figure(figsize=(9, 3))
    plt.plot(data[:, 1].astype(np.int32), data[:, 2], alpha=1, color=seaborn.color_palette()[0])
    # plt.plot(data[:, 1], smooth(data[:, 2]), color=seaborn.color_palette()[0])
    plt.title('Discriminator loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('discriminator_loss.svg', dpi=360,
                pad_inches=0., bbox_inches='tight')
    plt.close()


    return None

    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=4., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=3. + num_segments / 5., framerate=1. / 20., cars=cars)

    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    # load agents if resuming
    if resume_agents:
        for i, a in enumerate(agents):
            path = find_latest(resume_agents, 'agent_{}_*.pt'.format(i))
            print(f'Resuming agent {i} from path "{path}"')
            a.network.load_state_dict(torch.load(path))
            a.old_network.load_state_dict(torch.load(path))
            # a.network.cuda()
            # a.old_network.cuda()

            torch.save(a.network.state_dict(), 'agent_{}.pt'.format(i))

    # create discriminator
    # discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)

    # if resume:
    #     path = find_latest(resume, 'discriminator_[0-9]*.pt')
    #     print(f'Resuming discriminator from path "{path}"')
    #     discriminator.network.load_state_dict(torch.load(path))

    # create generator
    generators = {}
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if resume_generator:
        selected = glob.glob(os.path.join(resume_generator, 'generator_[0-9]*.pt'))
        for path in sorted(selected, key=lambda x: (len(x), x)):
            print(f'Loading generator from path "{path}"')
            generators[path] = torch.load(path)

    with torch.no_grad():
        # generated boards
        for path, gen in generators.items():
            print('Testing: {}'.format(path))
            generator.network.load_state_dict(gen)
            generated_boards = generator.generate(num_segments, eval_size)

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

            fig, ax = plt.subplots(1, 2)
            ax[0].hist(generated_winners[:, 1].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 1')
            ax[0].legend()
            ax[0].set_xlim(0., 1.)

            ax[1].hist(generated_winners[:, 2].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 2')
            ax[1].set_xlim(0., 1.)
            ax[1].legend()

            fig.text(0.5, 0.02, 'Win-rate of generated racetracks', ha='center')
            fig.text(0.02, 0.5, 'Number of racetracks', va='center', rotation='vertical')
            fig.savefig('plots/generated_{}.svg'.format(path.split('_')[-1].split('.')[0]),
                        bbox_inches='tight')
            plt.close(fig)

            tracks = generator.generate(num_segments, 16)
            game.reset(tracks)
            imgs = game.tracks_images(top_n=16)  # imgs = [16, 256, 256, 3]
            imgs = np.reshape(imgs, (4, 4, 256, 256, 3))
            imgs = np.transpose(imgs, (0, 2, 1, 3, 4))
            imgs = np.reshape(imgs, (4 * 256, 4 * 256, 3))

            fig, ax = plt.subplots(1, 1)
            ax.imshow(imgs)
            ax.set_axis_off()
            fig.savefig('plots/imgs_generated_{}.png'.format(path.split('_')[-1].split('.')[0]),
                        bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    main()
