import os
import queue
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
num_proc = 3


def play_games(balance_queue: mp.Queue, result_queue: mp.Queue):
    print('Starting generating games...')

    cars = [RaceCar(max_speed=60., acceleration=2., angle=60.),
            RaceCar(max_speed=60., acceleration=1., angle=90.)]
    game = Race(timeout=3. + num_segments / 1.5, framerate=1. / 20., cars=cars)

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions).to(device),
                       lr=5e-5, discount=0.99, eps=0.1)
              for _ in range(game.num_players)]

    # load agents if resuming
    for i, a in enumerate(agents):
        path = find_latest(learned_agents, 'agent_{}_*.pt'.format(i))
        a.network.load_state_dict(torch.load(path))

    results = {}
    while True:
        # generate boards
        boards = torch.empty((batch_size, num_segments, 2), dtype=torch.float, device=device)
        boards[:, :, 0].uniform_(-1., 1.)
        boards[:, :, 1].uniform_(0., 1.)

        # run agents to find who wins
        states, any_valid = game.reset(boards)

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
            results['summary/win_rates/player_{}'.format(p)] = (winners == p).float().mean().item()
        results['summary/invalid'] = (winners == -1).float().mean().item()

        result_queue.put(results)
        balance_queue.put((boards.to('cpu'), winners.to('cpu')))


def balance_batches(balance_queue: mp.Queue, batch_queue: mp.Queue):
    import queue

    print('Starting balancing batches...')
    assert batch_size % num_players == 0, "batch_size should be divisible by num_players"
    per_player = batch_size // num_players
    data = [queue.deque(maxlen=1024) for _ in range(num_players + 1)]

    while True:
        boards, winners = balance_queue.get()
        for board, winner in zip(boards, winners):
            data[winner.item() + 1].append(board)

        if all(len(q) >= per_player for q in data[1:]):
            batch = torch.empty((num_players, per_player, *boards.shape[1:]), dtype=torch.float)
            labels = torch.empty((num_players, per_player), dtype=torch.long)

            for p in range(num_players):
                for i in range(per_player):
                    batch[p, i] = data[p + 1].popleft()
                    labels[p, i] = p

            batch_queue.put((batch.view(-1, *batch.shape[2:]), labels.view(-1)))


def train(run_path, batch_queue: mp.Queue, result_queue: mp.Queue):
    print('Starting learning discriminator...')

    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-4)

    step = 0
    result = {}
    while True:
        boards, winners = batch_queue.get()
        dloss, dacc = discriminator.train(boards.to(device), winners.to(device))

        if step % 1000 == 0:
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, 'discriminator_{}.pt'.format(step)))
        step += 1

        result['summary/discriminator_loss'] = dloss
        result['summary/discriminator_accuracy'] = dacc

        result_queue.put(result)


def log_results(run_path, result_queue: mp.Queue):
    print('Starting logging results...')
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    step = 0
    while True:
        result = result_queue.get()
        for tag, data in result.items():
            summary_writer.add_scalar(tag, data, global_step=step)
        step += 1


def main():
    mp.set_start_method('spawn')

    run_path = find_next_run_dir('experiments')

    manager = mp.Manager()
    result_queue = manager.Queue(maxsize=1024)
    batch_queue = manager.Queue(maxsize=1024)
    balance_queue = manager.Queue(maxsize=4096)

    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(num_proc + 3)
    signal.signal(signal.SIGINT, sigint_handler)

    processes = [pool.apply_async(log_results, args=(run_path, result_queue)),
                 pool.apply_async(balance_batches, args=(balance_queue, batch_queue)),
                 pool.apply_async(train, args=(run_path, batch_queue, result_queue))]
    for pid in range(num_proc):
        processes.append(pool.apply_async(play_games, args=(balance_queue, result_queue)))

    try:
        while True:
            for p in processes:
                try:
                    p.get(timeout=1.)  # maybe one month will be enough
                except mp.TimeoutError:
                    pass
    except KeyboardInterrupt:
        print('Terminating pool...')
        pool.terminate()
    else:
        print('Closing pool...')
        pool.close()


if __name__ == '__main__':
    main()
