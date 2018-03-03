import random
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable

from game import Pacman, PytorchWrapper
from agents import A2CAgent
from networks import LSTMPolicy

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
    board, size, num_players = Pacman.from_str(DEFAULT_BOARD)
    game = PytorchWrapper(Pacman(size, num_players))
    agents = [A2CAgent(game.actions(),
                       LSTMPolicy(game.size, game.depth, game.actions()).cuda(), 0.99, beta=0.01)
              for _ in range(game.num_players)]
    params = sum((list(a.network.parameters()) for a in agents), [])
    optimizer = optim.Adam(params, lr=1e-4)
    board_grads = np.zeros_like(board, dtype=np.float32)
    for e in range(100000):
        total_rewards = [0] * num_players
        print('Starting episode {}'.format(e))
        optimizer.zero_grad()
        states = game.reset(board)
        for a in agents:
            a.network.reset_state()
        for _ in range(20):
            actions = [a.act(s) for a, s in zip(agents, states)]
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)
            for i, r in enumerate(rewards):
                total_rewards[i] += r
            if e % 100 == 0:
                print(' '.join(game.action_name(a) for a in actions),
                      ('[ ' + '{:6.3f} ' * num_players + ']').format(*rewards))
                print(game)
        print('Finished with scores:', ('[ ' + '{:6.3f} ' * num_players + ']').format(*total_rewards))
        print()
        for a in agents:
            a.learn()
        optimizer.step()
        # add grads
        board_grads += game.base_board.cpu().grad.data
        print('Grads: {:15.9f}'.format(np.sum(board_grads**2.)))


if __name__ == '__main__':
    main()
