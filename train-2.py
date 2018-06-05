import os
import random
import signal
import torch
from torch import optim
import torch.multiprocessing as mp
import numpy as np
from itertools import count
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest, device, one_hot

# learned_agents = os.path.join('learned')
resume = os.path.join('learned')
resume_segments = 128
num_players = 2
batch_size = 32
max_segments = 128
num_proc = trials = 4
latent = 64


def run_game(boards, winners, run_queue: mp.Queue, finish_queue: mp.Queue):
    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=3. + max_segments / 12., framerate=1. / 20., cars=cars)

    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    for i, a in enumerate(agents):
        path = find_latest(resume, 'agent_{}_*.pt'.format(i))
        print(f'Resuming agent {i} from path "{path}"')
        a.network.load_state_dict(torch.load(path))
        a.old_network.load_state_dict(torch.load(path))

    while True:
        # wait for signal to start
        run_queue.get(timeout=1000000.)

        states, any_valid = game.reset(boards.detach())
        game.record(random.randint(0, batch_size - 1))

        while any_valid and not game.finished():
            actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)

        for a in agents:
            a.reset()

        wins = one_hot(game.winners() + 1, num_classes=3)
        winners.copy_(wins)

        # put None to signal finished eval
        finish_queue.put(None)


def main():
    mp.set_start_method('spawn')

    manager = mp.Manager()
    run_queues = [manager.Queue(maxsize=1024) for _ in range(num_proc)]
    finish_queues = [manager.Queue(maxsize=1024) for _ in range(num_proc)]

    boards = torch.empty(batch_size, max_segments, 2, dtype=torch.float, device=device)
    winners = [torch.empty(batch_size, num_players + 1, dtype=torch.float, device=device)
               for _ in range(num_proc)]

    boards.share_memory_()
    for w in winners:
        w.share_memory_()

    run_path = find_next_run_dir('experiments')
    print(f'Running experiment {run_path}')

    # create game just for displaying
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=600., framerate=1. / 20., cars=cars)

    # create discriminator
    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)
    discriminator.async_optim(optim.Adam(discriminator.network.parameters(), lr=1e-5, weight_decay=0.0001))

    if resume:
        path = find_latest(resume, 'discriminator_*.pt')
        print(f'Resuming discriminator from path "{path}"')
        discriminator.network.load_state_dict(torch.load(path))

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)
    generator.async_optim(optim.Adam(generator.network.parameters(), lr=1e-5))  # dampening?

    if resume:
        path = find_latest(resume, 'generator_*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))

    # run a pool of threads
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(num_proc)
    signal.signal(signal.SIGINT, sigint_handler)

    processes = [pool.apply_async(run_game, args=(boards, winners[pid], run_queues[pid], finish_queues[pid]))
                 for pid in range(num_proc)]

    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    # some vars
    num_segments = max_segments if resume else 2

    try:
        for episode in count():
            print(f'Episode {episode}')

            # generate boards
            gen_boards = generator.generate(num_segments, batch_size)

            boards.copy_(gen_boards)
            # run agents to find who wins
            for trial in range(trials):
                run_queues[trial].put(None)

            winner = 0.
            for trial in range(trials):
                finish_queues[trial].get(timeout=100000.)
                winner += winners[trial]
            winner /= trials

            # discriminator calculate loss and perform backward pass
            dloss, dacc = discriminator.train(gen_boards.detach(), winner)

            summary_writer.add_scalar('summary/discriminator_loss', dloss, global_step=episode)
            summary_writer.add_scalar('summary/discriminator_accuracy', dacc, global_step=episode)

            # train generator
            pred_winners = discriminator.forward(gen_boards)
            gloss = generator.train(pred_winners)
            summary_writer.add_scalar('summary/generator_loss', gloss, global_step=episode)

            # log data
            for p in range(num_players):
                summary_writer.add_scalar(f'summary/win_rates/player_{p}', winner[:, p + 1].mean().item(),
                                          global_step=episode)
            summary_writer.add_scalar('summary/invalid', winner[:, 0].mean().item(), global_step=episode)

            if episode % 20 == 0:
                # save boards as images in tensorboard
                game.reset(gen_boards.detach())
                for i, img in enumerate(game.tracks_images(top_n=batch_size)):
                    summary_writer.add_image(f'summary/boards_{i}', img, global_step=episode)

            # save networks
            if episode % 200 == 0:
                torch.save(discriminator.network.state_dict(), os.path.join(run_path, f'discriminator_{episode}.pt'))
                torch.save(generator.network.state_dict(), os.path.join(run_path, f'generator_{episode}.pt'))
    except KeyboardInterrupt:
        print('Terminating pool...')
        pool.terminate()


if __name__ == '__main__':
    main()
