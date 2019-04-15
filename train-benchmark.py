import os
import gc
import time
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
from utils import find_next_run_dir, find_latest, one_hot, device

# learned_agents = None  # os.path.join('..', 'experiments', '013', 'run-4')
# learned_agents = os.path.join('..', 'experiments', '015', 'run-2')
learned_agents = None  # os.path.join('learned')
learned_discriminator = None  # os.path.join('experiments', 'run-16')
# learned_discriminator = os.path.join('experiments', 'run-6')
learned_generator = None  # os.path.join('experiments', 'run-16')
# learned_generator = os.path.join('experiments', 'run-3')
resume_segments = 128
# resume_segments = 128
num_players = 2
batch_size = 32 - 6
# batch_size = 32 - 4 * 6
max_segments = 128
num_proc = 2
trials = 6
# trials = 1
latent = 16
generator_batch_size = 64
generator_train_steps = 1


def predefined_tracks():
    tracks = torch.zeros(6, max_segments, 2, device=device)

    # zig-zag
    t0 = random.randint(0, 15)
    for i in range(t0, max_segments, 16):
        tracks[0, i: i + 16, 0] = 2. * ((i // 16) % 2) - 1.

    # one sharp turn
    t1 = random.randint(0, 100)
    tracks[1, t1: t1 + 20, 0] = 1.

    # S track
    t2 = random.randint(0, 70)
    tracks[2, t2: t2 + 12, 0] = 1.
    tracks[2, t2 + 30: t2 + 42, 0] = -1.

    # large U turn
    t3 = [random.randint(0, 5) for _ in range(7)]
    offs = [0, 10, 20, 40, 50, 70, 100]
    for o, t in zip(offs, t3):
        tracks[3, t + o: t + o + 5, 0] = 1.

    # small bumpy turn
    t4 = random.randint(0, 90)
    tracks[4, t4: t4 + 6, 0] = 1.
    tracks[4, t4 + 6: t4 + 18, 0] = -1.
    tracks[4, t4 + 18: t4 + 24, 0] = 1.

    # immediate turn
    t5 = random.randint(30, 60)
    tracks[5, :20, 0] = 1.
    tracks[5, -t5: -t5 + 20, 0] = -1.

    return tracks


def train(generator: RaceTrackGenerator, discriminator: RaceWinnerDiscriminator,
          agents: [PPOAgent], result_queue: mp.Queue, pid: int, run_path: str):
    print(f'{pid:3d} -- Training started...')

    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=60. * 10., framerate=1. / 20., cars=cars, observation_size=10)

    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    del game

    # load agents if resuming
    if learned_agents:
        for i, a in enumerate(agents):
            path = find_latest(learned_agents, 'agent_{}_*.pt'.format(i))
            print(f'Resuming agent {i} from path "{path}"')
            a.network.load_state_dict(torch.load(path))
            a.old_network.load_state_dict(torch.load(path))
            a.network.to(device)
            a.old_network.to(device)

    # create discriminator
    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)

    if learned_discriminator:
        path = find_latest(learned_discriminator, 'discriminator_[0-9]*.pt')
        print(f'Resuming discriminator from path "{path}"')
        discriminator.network.load_state_dict(torch.load(path))
        discriminator.network.to(device)

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if learned_generator:
        path = find_latest(learned_generator, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))
        generator.network.to(device)

    # setup optimizers
    generator.async_optim(optim.Adam(generator.network.parameters(), lr=1e-6))  # dampening?
    if learned_generator:
        path = find_latest(learned_generator, 'generator_opt_[0-9]*.pt')
        print(f'Resuming generator optimizer from path "{path}"')
        generator.optimizer.load_state_dict(torch.load(path))

    discriminator.async_optim(optim.Adam(discriminator.network.parameters(), lr=2e-6))
    if learned_discriminator:
        path = find_latest(learned_discriminator, 'discriminator_opt_[0-9]*.pt')
        print(f'Resuming discriminator optimizer from path "{path}"')
        discriminator.optimizer.load_state_dict(torch.load(path))

    for agent in agents:
        agent.async_optim(optim.Adam(agent.network.parameters(), lr=2e-6))  # , weight_decay=0.00001))  # 1e-5  default

    # params
    # num_segments = 2 if not learned_agents else resume_segments
    num_segments = resume_segments

    cars = [RaceCar(max_speed=60., acceleration=4., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=3. + num_segments / 5., framerate=1. / 20., cars=cars, observation_size=10)

    boards = generator.generate(num_segments, batch_size).detach()
    boards = torch.cat((boards, predefined_tracks()), dim=0)
    boards = torch.cat((boards, -boards), dim=0)  # mirror levels to train more robust agents
    rboards = boards.repeat(trials, 1, 1)

    print('rboards.shape == {}'.format(rboards.shape))

    game.record(0)

    t = 0.
    run_count = 100
    for i in range(run_count):
        print('\r[{:5d}/{:5d}]'.format(i + 1, run_count), end='')
        states, any_valid = game.reset(rboards)
        actions = torch.stack([a.act(s, training=False) for a, s in zip(agents, states)], dim=0)
        torch.cuda.synchronize()
        start = time.time()
        game.step(actions)
        torch.cuda.synchronize()
        t += time.time() - start
        for a in agents:
            a.reset()

    print()
    print('Average time: {:12.7f}s'.format(t / run_count))


def main():
    # mp.set_start_method('spawn')

    # run_path = find_next_run_dir('experiments')
    # print(f'Running experiment {run_path}')

    # manager = mp.Manager()
    # result_queue = manager.Queue(maxsize=1024)

    # run a pool of threads
    # sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    # pool = mp.Pool(num_proc + 1)
    # signal.signal(signal.SIGINT, sigint_handler)

    train(None, None, None, None, 0, '')

    # processes = [pool.apply_async(train, args=(None, None, None, result_queue, 0, ''))]
    #
    # try:
    #     while True:
    #         for p in processes:
    #             try:
    #                 p.get(timeout=1.)
    #             except mp.TimeoutError:
    #                 pass
    # except KeyboardInterrupt:
    #     print('Terminating pool...')
    #     pool.terminate()
    # else:
    #     print('Closing pool...')
    #     pool.close()


if __name__ == '__main__':
    main()
