from game import Pacman, PytorchWrapper
from generators import SimplePacmanGenerator
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
    # board, size, num_players = Pacman.from_str(DEFAULT_BOARD)

    size, num_players = (10, 10), 2
    board_generator = SimplePacmanGenerator(size, num_players)
    board = board_generator.generate()

    # create game
    game = PytorchWrapper(Pacman(size, num_players))

    # create agents with LSTM policy network
    agents = [A2CAgent(game.actions(),
                       LSTMPolicy(game.size, game.depth, game.actions()).cuda(),
                       0.99, beta=0.01)
              for _ in range(game.num_players)]

    for e in range(100000):
        total_rewards = [0] * num_players
        print('Starting episode {}'.format(e))
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

        # backward gradients to board generator with agents data about game
        board_generator.backward(agents)

        # update agent policies
        for a in agents:
            a.learn()


if __name__ == '__main__':
    main()
