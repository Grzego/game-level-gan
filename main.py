import os
import torch
import numpy as np
from game import Pacman, PytorchWrapper
from generators import SimplePacmanGenerator
from agents import A2CAgent
from networks import LSTMPolicy
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

    # create agents with LSTM policy network
    agents = [A2CAgent(game.actions,
                       cudify(LSTMPolicy(game.size, game.depth, game.actions)),
                       lr=1e-4, discount=0.99, beta=0.01)
              for _ in range(game.num_players)]

    run_path = find_next_run_dir('models')

    for e in range(100000):
        print()
        print('Starting episode {}'.format(e))

        boards = board_generator.generate(num_samples=batch_size)
        # TODO: make agents work on batches of levels
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

        # backward gradients to board generator with agents data about game
        board_generator.backward(agents)

        # update agent policies
        for a in agents:
            a.learn()

        # use gradients to train generator
        board_generator.train()

        if e % 1000 == 0:
            # save models
            for i, a in enumerate(agents):
                torch.save(a.network.state_dict(), os.path.join(run_path, 'agent_{}_{}.pt'.format(i, e)))
            torch.save(board_generator.generator.state_dict(), os.path.join(run_path, 'generator_{}.pt'.format(e)))


if __name__ == '__main__':
    main()
