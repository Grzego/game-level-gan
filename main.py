import os
import torch
import numpy as np
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import A2CAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest
from utils.pytorch_utils import device


resume = None  # os.path.join('experiments', 'run-6')


def main():
    # board, size, num_players = Pacman.from_str(DEFAULT_BOARD)

    latent, size, num_players = 128, (5, 5), 2
    # TODO: add race track generator
    track_generator = RaceTrackGenerator(latent, lr=1e-5)

    # create game
    batch_size = 1
    num_segments = 10
    game = Race(timeout=30., cars=[RaceCar(max_speed=100., acceleration=1., angle=45.)])#,
                                   #RaceCar(max_speed=80., acceleration=1., angle=60.)])

    # create discriminator for predicting winners
    # TODO: add race track winner discriminator
    # discriminator = RaceWinnerDiscriminator(track_generator.track_shape, num_players, lr=1e-5)

    # create agents with LSTM policy network
    agents = [A2CAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=1e-5, discount=0.9, beta=0.01)
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

    for e in range(epoch, 10000000):
        print()
        print('Starting episode {}'.format(e))

        # generate boards
        # boards = track_generator.generate(track_length=64, num_samples=batch_size)
        boards = torch.rand((batch_size, num_segments, 2), device=device)
        boards[:, :, 0] *= 2.
        boards[:, :, 0] -= 1.

        # run agents to find who wins
        total_rewards = np.zeros((batch_size, num_players))
        states = game.reset(boards)

        # TEST
        game.play()
        game.record_episode(os.path.join(run_path, 'videos', 'test'))
        return
        # -----

        while not game.finished():
            actions = torch.tensor([a.act(s) for a, s in zip(agents, states)], device=device)
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)
            for i, r in enumerate(rewards):
                total_rewards[:, i] += r
            if e % 100 == 0:
                print(' '.join(game.action_name(a) for a in actions[0]),
                      ('[ ' + '{:6.3f} ' * num_players + ']').format(*[r[0] for r in rewards]))
                print(game)
        print('Finished with scores:', ('[ ' + '{:6.3f} ' * num_players + ']').format(*total_rewards[0]))

        # update agent policies
        for i, a in enumerate(agents):
            aloss, mean_val = a.learn()
            summary_writer.add_scalar('summary/agent_{}/loss'.format(i), aloss, global_step=e)
            summary_writer.add_scalar('summary/agent_{}/mean_val'.format(i), mean_val, global_step=e)

        if e % 1000 == 0:
            # save models
            for i, a in enumerate(agents):
                torch.save(a.network.state_dict(), os.path.join(run_path, 'agent_{}_{}.pt'.format(i, e)))

        # discriminator calculate loss and perform backward pass
        winners = np.argmax(total_rewards, axis=1)
        winners = torch.tensor(winners, device=device)
        dloss, dacc = discriminator.train(boards.detach(), winners)

        summary_writer.add_scalar('summary/discriminator_loss', dloss, global_step=e)
        summary_writer.add_scalar('summary/discriminator_accuracy', dacc, global_step=e)

        # compute gradient for generator
        pred_winners = discriminator.forward(boards)
        gloss = track_generator.train(pred_winners)

        summary_writer.add_scalar('summary/generator_loss', gloss, global_step=e)

        if e % 100 == 0:
            # save boards as images in tensorboard
            # TODO: create image of race tracks and save them in tensorboard
            img_boards = None
            for i, img in enumerate(img_boards):
                summary_writer.add_image('summary/boards_{}'.format(i), img, global_step=e)

        if e % 1000 == 0:
            torch.save(track_generator.generator.state_dict(), os.path.join(run_path, 'generator_{}.pt'.format(e)))
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, 'discriminator_{}.pt'.format(e)))


if __name__ == '__main__':
    main()
