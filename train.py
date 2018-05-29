import os
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
from utils import find_next_run_dir, find_latest, device

# learned_agents = os.path.join('learned')
resume = None  # os.path.join('experiments', 'run-5')
num_players = 2
batch_size = 32
max_segments = 128
num_proc = 3
latent = 64


def train(generator: RaceTrackGenerator, discriminator: RaceWinnerDiscriminator,
          agents: [PPOAgent], result_queue: mp.Queue, pid: int, run_path: str):
    print(f'{pid:3d} -- Training started...')

    # setup optimizers
    generator.async_optim(optim.SGD(generator.network.parameters(), lr=1e-5, momentum=0.1))  # dampening?
    discriminator.async_optim(optim.RMSprop(discriminator.network.parameters(), lr=1e-5))
    for agent in agents:
        agent.async_optim(optim.Adam(agent.network.parameters(), lr=1e-4, weight_decay=0.0001))

    # flatten params
    generator.network.flatten_parameters()
    discriminator.network.flatten_parameters()
    for agent in agents:
        agent.network.flatten_parameters()
        agent.old_network.flatten_parameters()

    # create game
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=3., framerate=1. / 20., cars=cars)

    # params
    num_segments = 8
    finish_mean = 0.
    episode = -1

    result = {}
    while True:
        episode += 1
        if episode % 50 == 0:
            print(f'{pid:3d} -- episode {episode}')

        # generate boards
        boards = generator.generate(num_segments, batch_size)

        # run agents to find who wins
        states, any_valid = game.reset(boards.detach())
        game.record(random.randint(0, batch_size - 1))

        while any_valid and not game.finished():
            actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)

        # update agent policies
        for i, a in game.iterate_valid(agents):
            aloss, mean_val = a.learn()
            result[f'summary/agent_{i}/loss'] = aloss
            result[f'summary/agent_{i}/mean_val'] = mean_val

        cur_mean = game.finishes.float().mean().item()
        finish_mean = 0.9 * finish_mean + 0.1 * cur_mean
        result['summary/finishes'] = cur_mean

        if finish_mean > (0.75 + 0.1 * pid / (num_proc - 1)) and num_segments < max_segments:
            # increase number of segments and reset mean
            num_segments += 4
            finish_mean = 0.
            # change timeout so that players have time to finish race
            game.change_timeout(3. + num_segments / 5. / 1.5)
            print(f'{pid:3d} -- Increased number of segments to {num_segments}')

        # discriminator calculate loss and perform backward pass
        winners = game.winners()
        dloss, dacc = discriminator.train(boards.detach(), winners)
        result['summary/discriminator_loss'] = dloss
        result['summary/discriminator_accuracy'] = dacc

        # train generator
        pred_winners = discriminator.forward(boards)
        gloss = generator.train(pred_winners)
        result['summary/generator_loss'] = gloss

        # log data
        for p in range(num_players):
            result[f'summary/win_rates/player_{p}'] = (winners == p).float().mean().item()
        result['summary/invalid'] = (winners == -1).float().mean().item()

        # save episode
        if (episode - 50 * (pid + 1)) % (50 * num_proc) == 0:
            game.record_episode(os.path.join(run_path, 'videos', f'episode_{episode}'))
            # save boards as images in tensorboard
            for i, img in enumerate(game.tracks_images(top_n=batch_size)):
                result[f'summary/boards_{i}'] = img

        # save networks
        if (episode - 1000 * (pid + 1)) % (1000 * num_proc) == 0:
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, f'discriminator_{episode}.pt'))
            torch.save(generator.network.state_dict(), os.path.join(run_path, f'generator_{episode}.pt'))
            for i, a in enumerate(agents):
                torch.save(a.network.state_dict(), os.path.join(run_path, f'agent_{i}_{episode}.pt'))

        result_queue.put(result)
        result = {}


def log_results(run_path, result_queue: mp.Queue):
    from collections import defaultdict

    print('Starting logging results...')
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    steps = defaultdict(lambda: 0)
    while True:
        result = result_queue.get()
        for tag, data in result.items():
            if isinstance(data, np.ndarray):
                summary_writer.add_image(tag, data, global_step=steps[tag])
            else:
                summary_writer.add_scalar(tag, data, global_step=steps[tag])
            steps[tag] += 1


def main():
    mp.set_start_method('spawn')

    run_path = find_next_run_dir('experiments')

    manager = mp.Manager()
    result_queue = manager.Queue(maxsize=1024)

    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=2., angle=60.),
            RaceCar(max_speed=60., acceleration=1., angle=90.)]
    game = Race(timeout=3. + max_segments / 1.5, framerate=1. / 20., cars=cars)

    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions).to(device),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]
    for agent in agents:
        agent.network.share_memory()

    del game

    # load agents if resuming
    if resume:
        for i, a in enumerate(agents):
            path = find_latest(resume, 'agent_{}_*.pt'.format(i))
            a.network.load_state_dict(torch.load(path))

    # create discriminator
    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)
    discriminator.network.share_memory()

    if resume:
        path = find_latest(resume, 'discriminator_*.pt')
        discriminator.network.load_state_dict(torch.load(path))

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)
    generator.network.share_memory()

    if resume:
        path = find_latest(resume, 'generator_*.pt')
        generator.network.load_state_dict(torch.load(path))

    # run a pool of threads
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(num_proc + 1)
    signal.signal(signal.SIGINT, sigint_handler)

    processes = [pool.apply_async(log_results, args=(run_path, result_queue))]
    for pid in range(num_proc):
        processes.append(pool.apply_async(train, args=(generator, discriminator, agents, result_queue, pid, run_path)))

    try:
        while True:
            for p in processes:
                try:
                    p.get(timeout=1.)
                except mp.TimeoutError:
                    pass
    except KeyboardInterrupt:
        print('Terminating pool...')
        pool.terminate()
    # else:
    #     print('Closing pool...')
    #     pool.close()


if __name__ == '__main__':
    main()
