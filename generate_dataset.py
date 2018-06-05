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
resume = os.path.join('experiments', 'run-5')
dataset_size = 50000
num_players = 2
batch_size = 512
max_segments = 128
num_proc = 4
latent = 64
trials = 10


def main():
    print('-- Eval started...')

    # params
    num_segments = max_segments

    # create game
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=3. + num_segments / 8., framerate=1. / 20., cars=cars)

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
        path = find_latest(resume, 'generator_*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))

    time.clock()

    # create dataset
    with h5py.File('dataset.h5', 'w') as file:
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
                for trial in range(1, trials + 1):
                    eta = (time.clock() * (dataset_size - tracks.shape[0])) / (tracks.shape[0] + 1e-8)
                    print(f'\rSize: {tracks.shape[0]:6d}; trial: {trial:2d}; eta: {eta:20.3f}', end='')
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

                # append to dataset
                chunk = np.sum(not_invalid)
                if chunk > 0:
                    tracks.resize(tracks.shape[0] + chunk, axis=0)
                    winners.resize(winners.shape[0] + chunk, axis=0)

                    tracks[-chunk:] = boards.cpu().numpy()[not_invalid]
                    winners[-chunk:] = wins[not_invalid, :2]


if __name__ == '__main__':
    main()
