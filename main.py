import random
import numpy as np
from game import Pacman


DEFAULT_BOARD = np.array([[[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                          [[0, 0, 0, 0, 1], [0, 1, 0, 0, 0]]], dtype=np.int32)


def main():
    game = Pacman((2, 2), 2)
    game.reset(DEFAULT_BOARD)
    for _ in range(5):
        acts = [random.randint(0, 4), random.randint(0, 4)]
        states, rewards = game.step(acts)
        print(' '.join(game.action_name(a) for a in acts), rewards)
        print(game)


if __name__ == '__main__':
    main()
