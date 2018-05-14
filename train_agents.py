import os
import torch
import random
import numpy as np
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import PPOAgent
from networks import LSTMPolicy
from utils import find_next_run_dir, device


def main():
    # board, size, num_players = Pacman.from_str(DEFAULT_BOARD)

    num_players = 2

    # create game
    batch_size = 32
    num_segments = 1
    cars = [RaceCar(max_speed=60., acceleration=2., angle=60.),
            RaceCar(max_speed=60., acceleration=1., angle=90.)]
    game = Race(timeout=3. + num_segments / 1.5, framerate=1. / 20., cars=cars)

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions).to(device),
                       lr=5e-5, discount=0.99, eps=0.1)
              for _ in range(game.num_players)]

    run_path = find_next_run_dir('experiments')
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    epoch = 0
    finish_mean = 0.
    for e in range(epoch, 20000):
        print()
        print('Starting episode {}'.format(e))

        # generate boards
        boards = torch.empty((batch_size, num_segments, 2), dtype=torch.float, device=device)
        boards[:, :, 0].uniform_(-1., 1.)
        boards[:, :, 1].uniform_(0., 1.)

        # run agents to find who wins
        total_rewards = np.zeros((batch_size, num_players))
        states, any_valid = game.reset(boards.detach())
        game.record(random.randint(0, batch_size - 1))

        while any_valid and not game.finished():
            actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)
            for i, r in enumerate(rewards):
                total_rewards[:, i] += r

        print('Finished with rewards:', ('[ ' + '{:6.3f} ' * num_players + ']').format(*total_rewards[0]), end='')

        # TODO: large random tracks may be invalid making it impossible to add more segments
        cur_mean = game.finishes.float().mean().item()
        finish_mean = 0.9 * finish_mean + 0.1 * cur_mean
        print('; random finish mean: {:7.5f}'.format(finish_mean))

        summary_writer.add_scalar('summary/finishes', cur_mean, global_step=e)

        if e % 1000 == 0:
            for i, a in enumerate(agents):
                torch.save(a.network.state_dict(), os.path.join(run_path, 'agent_{}_{}.pt'.format(i, e)))

        if finish_mean > 0.8 and num_segments < 16:
            # increase number of segments and reset mean
            num_segments += 1
            finish_mean = 0.
            # change timeout so that players have time to finish race
            # TODO: use average race time here
            game.change_timeout(3. + num_segments / 1.5)
            print('{} -- Increased number of segments to {}'.format(e, num_segments))

        if e % 20 == 0:
            game.record_episode(os.path.join(run_path, 'videos', 'episode_{}'.format(e)))
            # save boards as images in tensorboard
            for i, img in enumerate(game.tracks_images(top_n=batch_size)):
                summary_writer.add_image('summary/boards_{}'.format(i), img, global_step=e)

        # update agent policies
        for i, a in game.iterate_valid(agents):
            aloss, mean_val = a.learn()
            summary_writer.add_scalar('summary/agent_{}/loss'.format(i), aloss, global_step=e)
            summary_writer.add_scalar('summary/agent_{}/mean_val'.format(i), mean_val, global_step=e)


if __name__ == '__main__':
    main()
