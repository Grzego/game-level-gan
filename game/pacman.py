import numpy as np

from .environment import MultiEnvironment


class Pacman(MultiEnvironment):
    """
    This version allows all players to be in the same place
    but reward is shared between them (if there is any)
    """

    def __init__(self, size: tuple, num_players: int):
        # size == [width, height]
        self.size = size
        self.fields = 3
        self.grid = np.zeros(size + (self.fields + num_players * 2,), dtype=np.int32)
        self.num_players = num_players
        self.players = None  # np.zeros((num_players, 2))
        self.moves = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int32)

    def reset(self, data):
        self.grid[:, :, :self.fields + self.num_players] = data
        positions = np.where(self.grid[:, :, self.fields: self.fields + self.num_players] == 1.)
        self.players = np.array(positions, dtype=np.int32).T[np.argsort(positions[2])][:, :2]

    def _not_blocked(self, positions):
        not_blocked = np.all(positions >= 0, axis=1)
        not_blocked &= np.all(positions < self.size, axis=1)
        pos_ins = positions[not_blocked]
        #                          VVVVVVVV because 0 layer means walls
        indices = np.r_[pos_ins.T, np.zeros((1, pos_ins.shape[0]), dtype=np.int32)]
        not_blocked[not_blocked] = self.grid[list(indices)] < 0.5
        return not_blocked

    def step(self, actions):
        new_pos = self.players + self.moves[actions]
        not_blocked = self._not_blocked(new_pos)

        reward = np.zeros((self.num_players,))
        if np.any(not_blocked):
            # remove players from grid
            positions = list(np.r_[self.players.T, np.arange(self.fields, self.fields + self.num_players,
                                                             dtype=np.int32)[None, :]])
            self.grid[positions] = 0

            # update players positions
            self.players[not_blocked] = new_pos[not_blocked]

            # add players to grid
            positions = list(np.r_[self.players.T, np.arange(self.fields, self.fields + self.num_players,
                                                             dtype=np.int32)[None, :]])
            self.grid[positions] = 1

            # calculate rewards
            #                             VVVVVVV because 1 layer means small-pellets and 2 large-pellets
            small = list(np.r_[self.players.T, np.ones((1, self.num_players), dtype=np.int32)])
            large = list(np.r_[self.players.T, 2 * np.ones((1, self.num_players), dtype=np.int32)])
            reward = 0.5 * self.grid[small] + self.grid[large]

            # split rewards between players
            uni, idx = np.unique(self.players, return_inverse=True, axis=0)
            freq = np.bincount(idx)[idx]
            reward /= freq.astype(np.float32)

            # remove pellets from grid
            self.grid[small] = 0
            self.grid[large] = 0

        # generate new state for each player
        return tuple(self.grid + np.eye(self.fields + 2 * self.num_players)[self.fields + self.num_players + player]
                     for player in range(self.num_players)), reward

    def actions(self):
        return 5  # [noop, up, down, left, right]

    def __repr__(self):
        def _field_repr(x, y):
            if self.grid[x, y, 0] == 1:
                return '█'
            if self.grid[x, y, 1] == 1:
                return '•'
            if self.grid[x, y, 2] == 1:
                return '♦'
            for p in range(self.num_players):
                if self.grid[x, y, self.fields + p] == 1:
                    return '{:1d}'.format(p)
            return ' '

        board = [[_field_repr(x, y)
                  for x in range(self.size[0])]
                 for y in range(self.size[1])]
        return '+' + '-' * self.size[0] + '+\n' + \
               '\n'.join('|' + ''.join(row) + '|' for row in board) + \
               '\n+' + '-' * self.size[0] + '+'

    @staticmethod
    def action_name(a):
        return ['noop', 'up', 'down', 'left', 'right'][a]
