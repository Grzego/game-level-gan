import os
import random
import torch
from torch import optim
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

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


def main():
    global generator, discriminator, agents, game
    seaborn.set()

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
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if resume_generator:
        path = find_latest(resume_generator, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))

    with torch.no_grad():
        # agents on own boards
        # own_boards = torch.zeros(eval_size, num_segments, 2, device=device)
        # for i in range(0, num_segments, 16):
        #     own_boards[:, i: i + 16, 0] = 2 * ((i // 16) % 2) - 1
        #
        # own_winners = 0.
        # for t in range(trials):
        #     states, any_valid = game.reset(own_boards)
        #     # game.record(0)
        #     print(f'[{t + 1:2d}/{trials:2d}] Own boards eval...')
        #     step = 0
        #     while any_valid and not game.finished():
        #         print(f'\r[{step:4d}]', end='')
        #         step += 1
        #         actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
        #         states, rewards = game.step(actions)
        #         for a, r in zip(agents, rewards):
        #             a.observe(r)
        #     print()
        #
        #     for a in agents:
        #         a.reset()
        #
        #     # game.record_episode('test')
        #
        #     own_winners += one_hot(game.winners() + 1, num_classes=num_players + 1).float()
        # own_winners /= trials
        #
        # print(own_winners.float().mean(0))
        # print(own_winners.float().std(0))
        #
        # plt.subplot(1, 2, 1)
        # plt.hist(own_winners[:, 1].float().cpu().numpy(), bins=trials + 1, label='player 1')
        # plt.xlim(0., 1.)
        # plt.subplot(1, 2, 2)
        # plt.hist(own_winners[:, 2].float().cpu().numpy(), bins=trials + 1, label='player 2')
        # plt.xlim(0., 1.)
        # # plt.show()
        # plt.savefig('own.png')
        # plt.close()

        # agents on random boards
        val = 1.
        random_boards = torch.zeros(eval_size, num_segments, 2, device=device)
        random_boards[:, :, 0].uniform_(-val, val)

        random_winners = 0.
        for t in range(trials):
            states, any_valid = game.reset(random_boards)
            game.record(0)
            print(f'[{t + 1:2d}/{trials:2d}] Random boards eval...')
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

            # game.record_episode('test')

            random_winners += one_hot(game.winners() + 1, num_classes=num_players + 1).float()
        random_winners /= trials

        print(random_winners.float().mean(0))
        print(random_winners.float().std(0))

        plt.subplot(1, 2, 1)
        plt.hist(random_winners[:, 1].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 1')
        plt.legend()
        plt.xlim(0., 1.)
        plt.subplot(1, 2, 2)
        plt.hist(random_winners[:, 2].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 2')
        plt.xlim(0., 1.)
        # plt.show()
        plt.legend()
        plt.savefig('random.png')
        plt.close()

        # generated dummy boards
        dummy_generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)
        generated_boards = dummy_generator.generate(num_segments, eval_size)

        generated_winners = 0.
        for t in range(trials):
            states, any_valid = game.reset(generated_boards)
            print(f'\r[{t + 1:2d}/{trials:2d}] Dummy boards eval...')
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
        plt.hist(generated_winners[:, 1].float().cpu().numpy(), bins=trials + 1, range=(0, 1),label='player 1')
        plt.legend()
        plt.xlim(0., 1.)
        plt.subplot(1, 2, 2)
        plt.hist(generated_winners[:, 2].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 2')
        plt.xlim(0., 1.)
        # plt.show()
        plt.legend()
        plt.savefig('dummy.png')
        plt.close()

        # generated boards
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


if __name__ == '__main__':
    main()
