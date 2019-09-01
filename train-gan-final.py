import os
import gc
import random
import signal
import torch
from torch import optim
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter

from games import race_game as game
from games import RaceConfig, predefined_tracks
from agents import PPOAgent
from generators import RaceTrackGenerator
from discriminators import RaceWinnerDiscriminator
from policies import LSTMPolicy
from utils import find_next_run_dir, find_latest, one_hot, device

learned_agents = os.path.join('learned')
learned_discriminator = None  # os.path.join('experiments', 'run-5')
learned_generator = None  # os.path.join('experiments', 'run-5')
resume_segments = 128
num_players = 2
batch_size = 32
max_segments = 128
num_proc = 1
trials = 6
latent = 16
observation_size = 18
generator_batch_size = 64
generator_train_steps = 1
beta = 0.


def train(result_queue: mp.Queue, pid: int, run_path: str):
    print(f'{pid:3d} -- Training started...')

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    # load agents if resuming
    for i, a in enumerate(agents):
        path = find_latest(learned_agents, 'agent_{}_*.pt'.format(i))
        print(f'Resuming agent {i} from path "{path}"')
        a.network.load_state_dict(torch.load(path))
        a.old_network.load_state_dict(torch.load(path))
        a.network.cuda()
        a.old_network.cuda()

    # create discriminator
    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)

    if learned_discriminator:
        path = find_latest(learned_discriminator, 'discriminator_[0-9]*.pt')
        print(f'Resuming discriminator from path "{path}"')
        discriminator.network.load_state_dict(torch.load(path))
        discriminator.network.cuda()

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if learned_generator:
        path = find_latest(learned_generator, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))
        generator.network.cuda()

    # setup optimizers
    generator.async_optim(optim.Adam(generator.network.parameters(), lr=1e-5, betas=(0.3, 0.9)))  # dampening?
    if learned_generator:
        path = find_latest(learned_generator, 'generator_opt_[0-9]*.pt')
        print(f'Resuming generator optimizer from path "{path}"')
        generator.optimizer.load_state_dict(torch.load(path))

    discriminator.async_optim(optim.Adam(discriminator.network.parameters(), lr=2e-5, betas=(0.5, 0.9)))
    if learned_discriminator:
        path = find_latest(learned_discriminator, 'discriminator_opt_[0-9]*.pt')
        print(f'Resuming discriminator optimizer from path "{path}"')
        discriminator.optimizer.load_state_dict(torch.load(path))

    for agent in agents:
        agent.async_optim(optim.Adam(agent.network.parameters(), lr=1e-5))  # 1e-5 start, 2e-6 later

    episode = -1
    result = {}
    while True:
        episode += 1
        if episode % 30 == 0:
            print(f'-- episode {episode}')

        # -- training discriminator
        boards = generator.generate(RaceConfig.max_segments, batch_size).detach()
        boards = torch.cat((boards, predefined_tracks()), dim=0)
        boards = torch.cat((boards, -boards), dim=0)  # mirror levels to train more robust discriminator
        rboards = boards.repeat(trials, 1, 1)

        states, any_valid = game.reset(rboards)
        game.record(0)

        # run agents to find who wins
        with torch.no_grad():
            while any_valid and not game.finished():
                actions = torch.stack([a.act(s, training=False) for a, s in zip(agents, states)], dim=0)
                states, rewards = game.step(actions)

        for a in agents:
            a.reset()

        cur_mean = game.finishes.float().mean().item()
        finish_mean = 0.9 * finish_mean + 0.1 * cur_mean
        result['game/finishes'] = cur_mean

        # discriminator calculate loss and perform backward pass
        winners = one_hot(game.winners() + 1, num_classes=game.num_players + 1)
        winners = winners.view(trials, -1, *winners.shape[1:]).float().mean(0)
        dloss, dacc = discriminator.train(boards.detach(), winners)
        result['discriminator/loss'] = dloss
        result['discriminator/accuracy'] = dacc

        # -- train generator
        for _ in range(generator_train_steps):
            generated = generator.generate(RaceConfig.max_segments, generator_batch_size)
            pred_winners = discriminator.forward(generated)
            gloss, galoss = generator.train(pred_winners, beta)
            result['generator/loss'] = gloss
            if galoss:
                result['generator/aux_loss'] = galoss

        # log data
        for p in range(num_players):
            result[f'game/win_rates/player_{p}'] = winners[:, p + 1].mean().item()
        result['game/invalid'] = winners[:, 0].mean().item()

        # save episode
        if pid == 0 and episode % 100 == 0:
            game.record_episode(os.path.join(run_path, 'videos', f'episode_{episode}'))
            # save boards as images in tensorboard
            # for i, img in enumerate(game.tracks_images(top_n=batch_size + 4 * 6)):
            for i, img in enumerate(game.tracks_images(top_n=batch_size)):
                result[f'game/boards_{i}'] = np.transpose(img, axes=(2, 0, 1))

        # save networks
        if pid == 0 and episode % 500 == 0:
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, f'discriminator_{episode}.pt'))
            torch.save(discriminator.optimizer.state_dict(), os.path.join(run_path, f'discriminator_opt_{episode}.pt'))
            torch.save(generator.network.state_dict(), os.path.join(run_path, f'generator_{episode}.pt'))
            torch.save(generator.optimizer.state_dict(), os.path.join(run_path, f'generator_opt_{episode}.pt'))

        result_queue.put(result)
        result.clear()


def log_results(run_path, result_queue: mp.Queue):
    from collections import defaultdict

    print('Starting logging results...')
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'),
                                   comment='Learning from zero. Discrete generator. Mirrored levels.')

    steps = defaultdict(lambda: 0)
    while True:
        result = result_queue.get()
        for tag, data in result.items():
            if isinstance(data, np.ndarray):
                summary_writer.add_image(tag, data, global_step=steps[tag])
            else:
                summary_writer.add_scalar(tag, data, global_step=steps[tag])
            steps[tag] += 1
        # -----
        if steps['gc-collect'] % 1000 == 0:
            gc.collect()
        steps['gc-collect'] += 1


def main():
    mp.set_start_method('spawn')

    run_path = find_next_run_dir('experiments')
    print(f'Running experiment {run_path}')


    processes = [pool.apply_async(log_results, args=(run_path, result_queue))]
    processes += [pool.apply_async(train, args=(None, None, None, result_queue, 0, run_path))]
    # train(None, None, None, result_queue, 0, run_path)
    # for pid in range(num_proc):
    #     processes.append(pool.apply_async(train, args=(generator, discriminator, agents, result_queue, pid, run_path)))

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
