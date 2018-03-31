import os
import torch
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter

from game import Pacman, PytorchWrapper
from generators import SimplePacmanGenerator
from agents import A2CAgent
from networks import LSTMPolicy, WinnerDiscriminator
from utils import find_next_run_dir
from utils.pytorch_utils import cudify

# DEFAULT_BOARD = np.array([[[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
#                           [[0, 0, 0, 0, 1], [0, 1, 0, 0, 0]]], dtype=np.int32)

DEFAULT_BOARD = \
    """
########
#2    s#
#### # #
#   S  #
#s ##  #
##  #  #
#S  # 1#
########
"""[1:-1]


def main():
    # board, size, num_players = Pacman.from_str(DEFAULT_BOARD)

    latent, size, num_players = 128, (5, 5), 2
    board_generator = SimplePacmanGenerator(latent, size, num_players, lr=1e-5)

    # create game
    batch_size = 32
    game = PytorchWrapper(Pacman(size, num_players, batch_size))

    # create discriminator for predicting winners
    discriminator = WinnerDiscriminator(game.size, game.grid_depth, num_players)

    # create agents with LSTM policy network
    agents = [A2CAgent(game.actions,
                       LSTMPolicy(game.size, game.depth, game.actions),
                       lr=1e-4, discount=0.99, beta=0.01)
              for _ in range(game.num_players)]

    run_path = find_next_run_dir('experiments')
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    for e in range(100000):
        print()
        print(f'Starting episode {e}')

        # generate boards
        boards = board_generator.generate(num_samples=batch_size)

        # run agents to find who wins
        total_rewards = np.zeros((batch_size, num_players))
        states = game.reset(boards)

        for _ in range(20):
            actions = np.array([a.act(s) for a, s in zip(agents, states)]).T
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
            summary_writer.add_scalar(f'summary/agent_{i}/loss', aloss, global_step=e)
            summary_writer.add_scalar(f'summary/agent_{i}/mean_val', mean_val, global_step=e)

        # board to PyTorch format
        p_boards = boards.permute(0, 3, 1, 2)

        # discriminator calculate loss and perform backward pass
        winners = np.argmax(total_rewards, axis=1)
        winners = Variable(cudify(torch.from_numpy(winners)))
        dloss = discriminator.train(p_boards.detach(), winners)

        # compute gradient for generator
        pred_winners = discriminator.forward(p_boards)
        gloss = board_generator.train(pred_winners)

        summary_writer.add_scalar('summary/discriminator_loss',  dloss, global_step=e)
        summary_writer.add_scalar('summary/generator_loss', gloss, global_step=e)

        if e % 100 == 0:
            # save boards as images in tensorboard
            n_boards = boards.data.cpu().numpy()  # [batch_size, height, width, depth]
            n_boards = n_boards[:3, ...]  # pick first 3 boards
            n_boards = np.argmax(n_boards, axis=-1)
            colors = np.array([[0., 0., 0.],  # empty
                               [1., 1., 1.],  # wall
                               [1., 0.64705882, 0.],  # small reward
                               [1., 0.84313725, 0.],  # large reward
                               [0., 1., 0.],  # 1 player
                               [0., 0., 1.],  # 2 player
                               [1., 0., 0.],  # 3 player
                               ])
            img_boards = colors[n_boards]
            for i, img in enumerate(img_boards):
                summary_writer.add_image(f'summary/boards_{i}', img, global_step=e)

        if e % 1000 == 0:
            # save models
            for i, a in enumerate(agents):
                torch.save(a.network.state_dict(), os.path.join(run_path, f'agent_{i}_{e}.pt'))
            torch.save(board_generator.generator.state_dict(), os.path.join(run_path, f'generator_{e}.pt'))


if __name__ == '__main__':
    main()
