import os
import gc
import random
import torch
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter
from collections import defaultdict

from games import race_game as game
from games import RaceConfig, predefined_tracks
from agents import PPOAgent
from policies import LSTMPolicy
from utils import find_next_run_dir, find_latest, device

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--resume-path', default=None, type=str)  # default='learned'
args = parser.parse_args()


def main():
    run_path = args.resume_path if args.resume_path else find_next_run_dir('experiments')
    print(f'Running experiment {run_path}')

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1)
              for _ in range(game.num_players)]

    # params
    finish_mean = 0.
    num_segments = 2
    episode = 0

    # load agents if resuming
    if args.resume_path:
        for i, a in enumerate(agents):
            path = find_latest(args.resume_path, 'agent_{}_*.pt'.format(i))
            print(f'Resuming agent {i} from path "{path}"')
            a.load(path)
        params = torch.load(find_latest(args.resume_path, 'params_*.pt'))
        finish_mean = params['finish_mean']
        num_segments = params['num_segments']
        episode = params['episode']

    # setup logger
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'), purge_step=episode)
    result = {}

    while True:
        if episode % 30 == 0:
            print(f'-- episode {episode}')

        # -- training agents
        boards = torch.zeros(args.batch_size, num_segments, 2, device=device)
        boards[:, :, 0].uniform_(-1., 1.)

        # concat predefined tracks
        boards = torch.cat((predefined_tracks()[:, :num_segments, :],
                            predefined_tracks()[:, :num_segments, :],
                            predefined_tracks()[:, :num_segments, :],
                            predefined_tracks()[:, :num_segments, :],
                            boards), dim=0)
        boards = torch.cat((boards, predefined_tracks()[:, :num_segments, :]), dim=0)
        boards = torch.cat((boards, -boards), dim=0)  # mirror levels to train more robust agents

        states, any_valid = game.reset(boards)
        game.record(0)

        # play games
        while any_valid and not game.finished():
            actions = torch.stack([a.act(s, training=True) for a, s in zip(agents, states)], dim=0)
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)

        # update agent networks
        for i, a in game.iterate_valid(agents):
            aloss, mean_val = a.learn()
            result[f'agents/agent_{i}/loss'] = aloss
            result[f'agents/agent_{i}/mean_val'] = mean_val

        for a in agents:
            a.reset()

        cur_mean = game.finishes.float().mean().item()
        finish_mean = 0.9 * finish_mean + 0.1 * cur_mean
        result['game/finishes'] = cur_mean

        if finish_mean >= 0.95 and num_segments < RaceConfig.max_segments:
            # increase number of segments and reset mean
            num_segments += 2
            finish_mean = 0.
            # change timeout so that players have time to finish race
            game.change_timeout(3. + num_segments / 5. + (num_segments <= 10) * 10.)
            print(f'{episode} -- Increased number of segments to {num_segments}')

        # save episode
        if episode % 100 == 0:
            game.record_episode(os.path.join(run_path, 'videos', f'episode_{episode}'))

        # save networks
        if episode % 500 == 0:
            for i, a in enumerate(agents):
                a.save(os.path.join(run_path, f'agent_{i}_{episode}.pt'))
            torch.save({
                'finish_mean': finish_mean,
                'num_segments': num_segments,
                'episode': episode,
            }, os.path.join(run_path, f'params_{episode}.pt'))

        # log results
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
