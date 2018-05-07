import os
import torch
import random
import numpy as np
from tensorboardX import SummaryWriter
from recordclass import recordclass

from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest

resume = None  # os.path.join('experiments', 'run-6')


def main():
    # board, size, num_players = Pacman.from_str(DEFAULT_BOARD)

    latent, num_players = 16, 2
    track_generator = RaceTrackGenerator(latent, lr=1e-6)

    # create game
    batch_size = 32
    num_segments = 1
    cars = [RaceCar(max_speed=60., acceleration=1., angle=45.),
            RaceCar(max_speed=45., acceleration=1., angle=60.)]
    game = Race(timeout=3., framerate=1./20., cars=cars)

    # create discriminator for predicting winners
    discriminator = RaceWinnerDiscriminator(num_players, lr=2e-5)

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1)
              for _ in range(game.num_players)]

    run_path = resume or find_next_run_dir('experiments')
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    epoch = 0
    # load agents if resuming
    if resume:
        for i, a in enumerate(agents):
            path = find_latest(resume, 'agent_{}_*.pt'.format(i))
            a.network.load_state_dict(torch.load(path))
            epoch = int(path.split('_')[-1].split('.')[0])

    finishes = recordclass('FinishMeans', 'generated random')(0., 0.)
    for e in range(epoch, 10000000):
        print()
        print('Starting episode {}'.format(e))

        # generate boards
        random_split = 8 if num_segments >= 12 else batch_size // 2
        generated_boards = track_generator.generate(track_length=num_segments, num_samples=batch_size - random_split)
        random_boards = generated_boards.new_empty((random_split, num_segments, 2))
        random_boards[:, :, 0].uniform_(-1., 1.)
        random_boards[:, :, 1].uniform_(0., 1.)
        boards = torch.cat((generated_boards, random_boards))

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
        finish_gen = game.finishes[: -random_split].float().mean().item()
        finish_rand = game.finishes[-random_split:].float().mean().item()
        finishes.generated = 0.9 * finishes.generated + 0.1 * finish_gen
        finishes.random = 0.9 * finishes.random + 0.1 * finish_rand
        print('; generated mean: {:7.5f}; random mean: {:7.5f}'.format(*finishes))

        summary_writer.add_scalar('summary/finishes/generated', finish_gen, global_step=e)
        summary_writer.add_scalar('summary/finishes/random', finish_rand, global_step=e)

        if finishes.generated > 0.8 and finishes.random > 0.8 and num_segments < 64:
            # increase number of segments and reset mean
            num_segments += 1
            finishes.generated, finishes.random = 0., 0.
            # change timeout so that players have time to finish race
            # TODO: use average race time here
            game.change_timeout(3. + num_segments / 1.5)
            print('{} -- Increased number of segments to {}'.format(e, num_segments))

        # update agent policies
        for i, a in game.iterate_valid(agents):
            aloss, mean_val = a.learn()
            summary_writer.add_scalar('summary/agent_{}/loss'.format(i), aloss, global_step=e)
            summary_writer.add_scalar('summary/agent_{}/mean_val'.format(i), mean_val, global_step=e)

        if e % 1000 == 0:
            # save models
            for i, a in enumerate(agents):
                torch.save(a.network.state_dict(), os.path.join(run_path, 'agent_{}_{}.pt'.format(i, e)))

        # discriminator calculate loss and perform backward pass
        winners = game.winners()
        for p in range(num_players):
            summary_writer.add_scalar('summary/win_rates/player_{}'.format(p),
                                      (winners == p).float().mean(), global_step=e)
        summary_writer.add_scalar('summary/invalid', (winners == -1).float().mean(), global_step=e)

        dloss, dacc = discriminator.train(boards.detach(), winners)

        summary_writer.add_scalar('summary/discriminator_loss', dloss, global_step=e)
        summary_writer.add_scalar('summary/discriminator_accuracy', dacc, global_step=e)

        # compute gradient for generator
        # if num_segments >= 8:
        pred_winners = discriminator.forward(generated_boards)
        gloss = track_generator.train(pred_winners)

        summary_writer.add_scalar('summary/generator_loss', gloss, global_step=e)

        if e % 20 == 0:
            game.record_episode(os.path.join(run_path, 'videos', 'episode_{}'.format(e)))
            # save boards as images in tensorboard
            for i, img in enumerate(game.tracks_images(top_n=batch_size)):
                summary_writer.add_image('summary/boards_{}'.format(i), img, global_step=e)

        if e % 1000 == 0:  # and num_segments >= 8:
            torch.save(track_generator.generator.state_dict(), os.path.join(run_path, 'generator_{}.pt'.format(e)))
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, 'discriminator_{}.pt'.format(e)))


if __name__ == '__main__':
    main()
