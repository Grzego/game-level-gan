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
from utils import find_next_run_dir, find_latest, one_hot

learned_agents = os.path.join('learned')
learned_discriminator = os.path.join('experiments', 'run-11')
learned_generator = os.path.join('experiments', 'run-2')
resume_segments = 128
num_players = 2
batch_size = 32
max_segments = 128
num_proc = 2
trials = 2
# latent = 4
latent = 64


def train(generator: RaceTrackGenerator, discriminator: RaceWinnerDiscriminator,
          agents: [PPOAgent], result_queue: mp.Queue, pid: int, run_path: str):
    print(f'{pid:3d} -- Training started...')

    # setup optimizers
    generator.async_optim(optim.Adam(generator.network.parameters(), lr=1e-5))  # dampening?
    discriminator.async_optim(optim.Adam(discriminator.network.parameters(), lr=1e-5, weight_decay=0.0001))
    for agent in agents:
        agent.async_optim(optim.Adam(agent.network.parameters(), lr=1e-6, weight_decay=0.0001))  # 1e-5  default

    # params
    num_segments = 2 if not learned_agents else resume_segments
    finish_mean = 0.
    episode = -1

    # create game
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=3. + num_segments / 12., framerate=1. / 20., cars=cars)

    result = {}
    while True:
        episode += 1
        if episode % 30 == 0:
            print(f'{pid:3d} -- episode {episode}')

        # generate boards
        boards = generator.generate(num_segments, batch_size)
        boards = torch.cat((boards, -boards), dim=0)  # add mirrored levels

        # run agents to find who wins
        rboards = boards.detach().repeat(trials, 1, 1)
        states, any_valid = game.reset(rboards)
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
        for a in agents:
            a.reset()

        cur_mean = game.finishes.float().mean().item()
        finish_mean = 0.9 * finish_mean + 0.1 * cur_mean
        result['summary/finishes'] = cur_mean

        if finish_mean >= 0.9 and num_segments < max_segments:
            # increase number of segments and reset mean
            num_segments += 2
            finish_mean = 0.
            # change timeout so that players have time to finish race
            game.change_timeout(3. + num_segments / 12.)
            print(f'{pid:3d} -- Increased number of segments to {num_segments}')

        # discriminator calculate loss and perform backward pass
        winners = one_hot(game.winners() + 1, num_classes=num_players + 1)
        winners = winners.view(trials, -1, *winners.shape[1:]).float().mean(0)
        dloss, dacc = discriminator.train(boards.detach(), winners)
        result['summary/discriminator_loss'] = dloss
        result['summary/discriminator_accuracy'] = dacc

        # train generator
        # pred_winners = discriminator.forward(boards)
        # gloss = generator.train(pred_winners)
        # result['summary/generator_loss'] = gloss

        # log data
        for p in range(num_players):
            result[f'summary/win_rates/player_{p}'] = winners[:, p + 1].mean().item()
        result['summary/invalid'] = winners[:, 0].mean().item()

        # save episode
        if pid == 0 and episode % 20 == 0:
            game.record_episode(os.path.join(run_path, 'videos', f'episode_{episode}'))
            # save boards as images in tensorboard
            for i, img in enumerate(game.tracks_images(top_n=batch_size)):
                result[f'summary/boards_{i}'] = img

        # save networks
        if pid == 0 and episode % 200 == 0:
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
    print(f'Running experiment {run_path}')

    manager = mp.Manager()
    result_queue = manager.Queue(maxsize=1024)

    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=60. * 10., framerate=1. / 20., cars=cars)

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

    # create discriminator
    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)

    if learned_discriminator:
        path = find_latest(learned_discriminator, 'discriminator_*.pt')
        print(f'Resuming discriminator from path "{path}"')
        discriminator.network.load_state_dict(torch.load(path))

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if learned_generator:
        path = find_latest(learned_generator, 'generator_*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))

    # train(generator, discriminator, agents, result_queue, 0, run_path)

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
