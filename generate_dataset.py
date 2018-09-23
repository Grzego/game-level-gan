import os
import time
import h5py
import random
import signal
import torch
from torch import optim
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest

# learned_agents = os.path.join('learned')
resume = os.path.join('experiments', 'run-4')
dataset_size = 50000
num_players = 2
batch_size = 500
max_segments = 128
num_proc = 4
latent = 8
trials = 10


def main():
    print('-- Generating dataset started...')

    # params
    num_segments = max_segments

    # create game
    cars = [RaceCar(max_speed=60., acceleration=4., angle=60.),
            RaceCar(max_speed=60., acceleration=2., angle=90.)]
    game = Race(timeout=3. + num_segments / 20., framerate=1. / 20., cars=cars)

    # create agents
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    # load agents
    if resume:
        for i, a in enumerate(agents):
            path = find_latest(resume, 'agent_{}_*.pt'.format(i))
            print(f'Resuming agent {i} from path "{path}"')
            # TODO: move loading to PPO class
            a.network.load_state_dict(torch.load(path))
            a.old_network.load_state_dict(torch.load(path))

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if resume:
        path = find_latest(resume, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))

    time.clock()

    # create dataset
    with h5py.File('dataset-4.h5', 'w') as file:
        tracks = file.create_dataset('tracks', shape=(0, 128, 2), maxshape=(None, 128, 2), dtype=np.float32)
        winners = file.create_dataset('winners', shape=(0, 2), maxshape=(None, 2), dtype=np.float32)

        with torch.no_grad():
            while tracks.shape[0] < dataset_size:
                # generate boards
                boards = generator.generate(num_segments, batch_size)
                # boards = torch.empty(batch_size, num_segments, 2, device='cuda')
                # boards[:, :, 0].uniform_(-1., 1.)
                # boards[:, :, 1].zero_()

                not_invalid = None
                wins = 0
                # run agents to find who wins
                eta = (time.clock() * (dataset_size - tracks.shape[0])) / (tracks.shape[0] + 1e-8)
                for trial in range(1, trials + 1):
                    print(f'Size: {tracks.shape[0]:6d}; trial: {trial:2d}; eta: {eta:20.3f}')
                    states, any_valid = game.reset(boards.detach())

                    while any_valid and not game.finished():
                        actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
                        states, rewards = game.step(actions)
                        for a, r in zip(agents, rewards):
                            a.observe(r)

                    for a in agents:
                        a.reset()

                    current_winners = game.winners().cpu().numpy()
                    not_invalid = current_winners != -1
                    wins += np.eye(3, dtype=np.float32)[current_winners, :2]

                wins /= trials

                pl1 = wins[:, 0] >= 0.8  # player1 wins
                pl2 = wins[:, 1] >= 0.8  # player2 wins

                print(f'Stats: {np.mean(pl1):5.4} {np.mean(pl2):5.4} {1. - np.mean(pl1) - np.mean(pl2):5.4}')

                low_win = int(min(np.sum(pl1), np.sum(pl2)))
                if low_win > 0:
                    boards_cpu = boards.cpu().numpy()
                    boards_to_save = np.concatenate((boards_cpu[pl1, :, :][:low_win, :, :],
                                                     boards_cpu[pl2, :, :][:low_win, :, :]), axis=0)
                    wins_to_save = np.concatenate((wins[pl1, :][:low_win, :2],
                                                   wins[pl2, :][:low_win, :2]), axis=0)

                    tracks.resize(tracks.shape[0] + low_win * 2, axis=0)
                    winners.resize(winners.shape[0] + low_win * 2, axis=0)

                    tracks[-low_win * 2:] = boards_to_save
                    winners[-low_win * 2:] = wins_to_save

                # # append to dataset
                # chunk = np.sum(not_invalid)
                # if chunk > 0:
                #     tracks.resize(tracks.shape[0] + chunk, axis=0)
                #     winners.resize(winners.shape[0] + chunk, axis=0)
                #
                #     tracks[-chunk:] = boards.cpu().numpy()[not_invalid]
                #     winners[-chunk:] = wins[not_invalid, :2]


if __name__ == '__main__':
    main()
