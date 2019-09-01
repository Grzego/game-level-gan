import os
import gc
import torch
import numpy as np
from tensorboardX import SummaryWriter

from games import race_game as game
from games import RaceConfig, predefined_tracks
from agents import PPOAgent
from generators import RaceTrackGenerator
from discriminators import RaceWinnerDiscriminator
from policies import LSTMPolicy
from utils import find_next_run_dir, find_latest, one_hot, device

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--resume-path', default=None, type=str)  # default='learned'
parser.add_argument('--trials', default=6, type=int,
                    help='Number of times we simulate game to determine a winner.')
parser.add_argument('--agents', default='learned', type=str,
                    help='Path to trained agents.')
parser.add_argument('--latent', default=16, type=int,
                    help='Dimensionality of latent vector.')
parser.add_argument('--generator-batch-size', '-gbs', default=64, type=int)
parser.add_argument('--generator-train-steps', '-gts', default=1, type=int)
parser.add_argument('--generator-beta', '-b', default=0., type=float,
                    help='Auxiliary loss scale.')
args = parser.parse_args()


def main():
    run_path = args.resume_path if args.resume_path else find_next_run_dir('experiments')
    print(f'Running experiment {run_path}')

    episode = 0

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1)
              for _ in range(game.num_players)]

    # load agents if resuming
    for i, a in enumerate(agents):
        path = find_latest(args.agents, 'agent_{}_*.pt'.format(i))
        print(f'Resuming agent {i} from path "{path}"')
        a.load(path)

    # create discriminator
    discriminator = RaceWinnerDiscriminator(game.num_players, lr=1e-5, betas=(0.5, 0.9))

    # create generator
    generator = RaceTrackGenerator(args.latent, lr=1e-5, betas=(0.3, 0.9))

    if args.resume_path:
        path = find_latest(args.resume_path, 'discriminator_[0-9]*.pt')
        print(f'Resuming discriminator from path "{path}"')
        discriminator.load(path)

        path = find_latest(args.resume_path, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.load(path)

        params = torch.load(find_latest(args.resume_path, 'params_[0-9]*.pt'))
        episode = params['episode']

    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'), purge_step=episode)
    result = {}

    while True:
        if episode % 30 == 0:
            print(f'-- episode {episode}')

        # -- training discriminator
        boards = generator.generate(RaceConfig.max_segments, args.batch_size).detach()
        boards = torch.cat((boards, predefined_tracks()), dim=0)
        boards = torch.cat((boards, -boards), dim=0)  # mirror levels to train more robust discriminator
        rboards = boards.repeat(args.trials, 1, 1)

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
        winners = winners.view(args.trials, -1, *winners.shape[1:]).float().mean(0)
        dloss, dacc = discriminator.train(boards.detach(), winners)
        result['discriminator/loss'] = dloss
        result['discriminator/accuracy'] = dacc

        # -- train generator
        for _ in range(args.generator_train_steps):
            generated = generator.generate(RaceConfig.max_segments, args.generator_batch_size)
            pred_winners = discriminator.forward(generated)
            gloss, galoss = generator.train(pred_winners, args.beta)
            result['generator/loss'] = gloss
            if galoss:
                result['generator/aux_loss'] = galoss

        # log data
        for p in range(game.num_players):
            result[f'game/win_rates/player_{p}'] = winners[:, p + 1].mean().item()
        result['game/invalid'] = winners[:, 0].mean().item()

        # save episode
        if episode % 100 == 0:
            game.record_episode(os.path.join(run_path, 'videos', f'episode_{episode}'))
            # save boards as images in tensorboard
            # for i, img in enumerate(game.tracks_images(top_n=batch_size + 4 * 6)):
            for i, img in enumerate(game.tracks_images(top_n=args.batch_size)):
                result[f'game/boards_{i}'] = np.transpose(img, axes=(2, 0, 1))

        # save networks
        if episode % 500 == 0:
            discriminator.save(os.path.join(run_path, f'discriminator_{episode}.pt'))
            generator.save(os.path.join(run_path, f'generator_{episode}.pt'))

        # save data to tensorboard
        for tag, data in result.items():
            if isinstance(data, np.ndarray):
                summary_writer.add_image(tag, data, global_step=episode)
            else:
                summary_writer.add_scalar(tag, data, global_step=episode)
        # -----
        if episode % 1000 == 0:
            gc.collect()
        result.clear()
        episode += 1


if __name__ == '__main__':
    main()
