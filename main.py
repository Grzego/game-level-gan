import os
import torch
import numpy as np
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest

resume = None  # os.path.join('experiments', 'run-6')


def main():
    # board, size, num_players = Pacman.from_str(DEFAULT_BOARD)

    latent, num_players = 16, 2
    # TODO: add race track generator
    track_generator = RaceTrackGenerator(latent, lr=1e-5)

    # create game
    batch_size = 32
    num_segments = 1
    cars = [RaceCar(max_speed=100., acceleration=1., angle=45.),
            RaceCar(max_speed=80., acceleration=1., angle=60.)]
    game = Race(timeout=5., framerate=1./20., cars=cars)

    # create discriminator for predicting winners
    # TODO: add race track winner discriminator
    discriminator = RaceWinnerDiscriminator(num_players, lr=2e-5)

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=1e-4, discount=0.99, beta=0.01)
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

    finish_mean = 0.
    for e in range(epoch, 10000000):
        print()
        print('Starting episode {}'.format(e))

        # generate boards
        boards = track_generator.generate(track_length=num_segments, num_samples=batch_size)
        # boards = torch.rand((batch_size, num_segments, 2), device=device)
        # boards[:, :, 0] *= 2.
        # boards[:, :, 0] -= 1.

        # run agents to find who wins
        total_rewards = np.zeros((batch_size, num_players))
        states, any_valid = game.reset(boards.detach())

        while any_valid and not game.finished():
            actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)
            for i, r in enumerate(rewards):
                total_rewards[:, i] += r
        finish_mean = 0.9 * finish_mean + 0.1 * game.finishes.float().mean().item()
        print('Finished with rewards:', ('[ ' + '{:6.3f} ' * num_players + ']').format(*total_rewards[0]),
              '; finish mean: {:7.5f}'.format(finish_mean))

        if finish_mean > 0.9:
            # increase number of segments and reset mean
            num_segments += 1
            finish_mean = 0.
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
        if num_segments >= 4:
            pred_winners = discriminator.forward(boards)
            gloss = track_generator.train(pred_winners)

            summary_writer.add_scalar('summary/generator_loss', gloss, global_step=e)

        if e % 20 == 0:
            game.record_episode(os.path.join(run_path, 'videos', 'episode_{}'.format(e)))
            # save boards as images in tensorboard
            for i, img in enumerate(game.tracks_images(top_n=3)):
                summary_writer.add_image('summary/boards_{}'.format(i), img, global_step=e)

        if e % 1000 == 0 and num_segments >= 4:
            torch.save(track_generator.generator.state_dict(), os.path.join(run_path, 'generator_{}.pt'.format(e)))
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, 'discriminator_{}.pt'.format(e)))


if __name__ == '__main__':
    main()
