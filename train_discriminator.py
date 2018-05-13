import os
import random
import signal
import numpy as np
import torch
from torch import optim
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import PPOAgent
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest, device

learned_agents = os.path.join('learned')
num_players = 2
batch_size = 32
num_segments = 16
num_proc = 2


def train(discriminator: RaceWinnerDiscriminator, run_path: str, result_queue: mp.Queue, pid: int):
    discriminator.async_optim(optim.Adam(discriminator.network.parameters(), lr=1e-4))

    cars = [RaceCar(max_speed=60., acceleration=2., angle=60.),
            RaceCar(max_speed=60., acceleration=1., angle=90.)]
    game = Race(timeout=3. + num_segments / 1.5, framerate=1. / 20., cars=cars)

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1)
              for _ in range(game.num_players)]

    # load agents if resuming
    for i, a in enumerate(agents):
        path = find_latest(learned_agents, 'agent_{}_*.pt'.format(i))
        a.network.load_state_dict(torch.load(path))

    results = {}
    for e in range(0, 10000000, pid):
        # generate boards
        boards = torch.empty((batch_size, num_segments, 2), dtype=torch.float, device=device)
        boards[:, :, 0].uniform_(-1., 1.)
        boards[:, :, 1].uniform_(0., 1.)

        # run agents to find who wins
        states, any_valid = game.reset(boards.detach())
        game.record(random.randint(0, batch_size - 1))

        while any_valid and not game.finished():
            actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)

        for a in agents:
            a.reset()

        # discriminator calculate loss and perform backward pass
        winners = game.winners()
        for p in range(num_players):
            results['summary/win_rates/player_{}'.format(p)] = (winners == p).float().mean()
        results['summary/invalid'] = (winners == -1).float().mean()

        dloss, dacc = discriminator.train(boards.detach(), winners)

        results['summary/discriminator_loss'] = dloss
        results['summary/discriminator_accuracy'] = dacc

        if e % 1000 == 0:  # and num_segments >= 8:
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, 'discriminator_{}.pt'.format(e)))

        result_queue.put(results)


def log_results(run_path, result_queue: mp.Queue):
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    step = 0
    while True:
        result = result_queue.get()
        for tag, data in result.items():
            summary_writer.add_scalar(tag, data, global_step=step)
        step += 1


def main():
    mp.set_start_method('spawn')

    # create discriminator for predicting winners
    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-4, asynchronous=True)
    discriminator.network.share_memory()

    run_path = find_next_run_dir('experiments')

    manager = mp.Manager()
    result_queue = manager.Queue(maxsize=1024)

    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(num_proc + 1)
    signal.signal(signal.SIGINT, sigint_handler)

    processes = []
    for pid in range(num_proc):
        processes.append(pool.apply_async(train, args=(discriminator, run_path, result_queue, pid)))
    processes.append(pool.apply_async(log_results, args=(run_path, result_queue)))

    try:
        for p in processes:
            p.get(timeout=31 * 24 * 60 * 60)  # maybe one month will be enough
    except KeyboardInterrupt:
        print('Terminating pool...')
        pool.terminate()
    else:
        print('Closing pool...')
        pool.close()


if __name__ == '__main__':
    main()
