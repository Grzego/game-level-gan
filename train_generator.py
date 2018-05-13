import os
import torch
import random
import numpy as np
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest, device

learned_agents = os.path.join('learned')
learned_discriminator = os.path.join('learned')


def main():
    # board, size, num_players = Pacman.from_str(DEFAULT_BOARD)

    latent, num_players = 64, 2
    track_generator = RaceTrackGenerator(latent, lr=1e-5)

    # create game
    batch_size = 32
    num_segments = 1 if learned_agents is None else 16
    cars = [RaceCar(max_speed=60., acceleration=2., angle=60.),
            RaceCar(max_speed=60., acceleration=1., angle=90.)]
    game = Race(timeout=3. + num_segments / 1.5, framerate=1. / 20., cars=cars)

    # create discriminator for predicting winners
    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-4)

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1)
              for _ in range(game.num_players)]

    run_path = find_next_run_dir('experiments')
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    epoch = 0
    # load agents if resuming
    for i, a in enumerate(agents):
        path = find_latest(learned_agents, 'agent_{}_*.pt'.format(i))
        a.network.load_state_dict(torch.load(path))
        epoch = int(path.split('_')[-1].split('.')[0])

    disc_path = find_latest(learned_discriminator, 'discriminator_*.pt')
    discriminator.network.load_state_dict(torch.load(disc_path))

    finish_mean = 0.
    for e in range(epoch, 10000000):
        print()
        print('Starting episode {}'.format(e))

        # generate boards
        boards = track_generator.generate(track_length=num_segments, num_samples=batch_size)

        # run agents to find who wins
        # total_rewards = np.zeros((batch_size, num_players))
        # states, any_valid = game.reset(boards.detach())
        # game.record(random.randint(0, batch_size - 1))
        #
        # while any_valid and not game.finished():
        #     actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
        #     states, rewards = game.step(actions)
        #     for a, r in zip(agents, rewards):
        #         a.observe(r)
        #     for i, r in enumerate(rewards):
        #         total_rewards[:, i] += r
        #
        # print('Finished with rewards:', ('[ ' + '{:6.3f} ' * num_players + ']').format(*total_rewards[0]), end='')
        #
        # # TODO: large random tracks may be invalid making it impossible to add more segments
        # cur_mean = game.finishes.float().mean().item()
        # finish_mean = 0.9 * finish_mean + 0.1 * cur_mean
        # print('; random finish mean: {:7.5f}'.format(finish_mean))
        #
        # summary_writer.add_scalar('summary/finishes', cur_mean, global_step=e)
        #
        # if e % 20 == 0:
        #     game.record_episode(os.path.join(run_path, 'videos', 'episode_{}'.format(e)))
        #     # save boards as images in tensorboard
        #     for i, img in enumerate(game.tracks_images(top_n=batch_size)):
        #         summary_writer.add_image('summary/boards_{}'.format(i), img, global_step=e)
        #
        # for a in agents:
        #     a.reset()
        #
        # #  game stats
        # winners = game.winners()
        # for p in range(num_players):
        #     summary_writer.add_scalar('summary/win_rates/player_{}'.format(p),
        #                               (winners == p).float().mean(), global_step=e)
        # summary_writer.add_scalar('summary/invalid', (winners == -1).float().mean(), global_step=e)

        pred_winners = discriminator.forward(boards)
        gloss = track_generator.train(pred_winners)

        summary_writer.add_scalar('summary/generator_loss', gloss, global_step=e)

        if e % 1000 == 0:
            torch.save(track_generator.generator.state_dict(), os.path.join(run_path, 'generator_{}.pt'.format(e)))


if __name__ == '__main__':
    main()
