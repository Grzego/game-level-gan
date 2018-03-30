import numpy as np

from .environment import MultiEnvironment


class Pacman(MultiEnvironment):
    """
    This version allows all players to be in the same place
    but reward is shared between them (if there is any)
    """

    def __init__(self, size: tuple, num_players: int, batch_size=32):
        # size == [width, height]
        self.size = size
        self.fields = 4
        self.depth = self.fields + num_players * 2
        self.grid_depth = self.fields + num_players
        self.grid = np.zeros((batch_size,) + size + (self.fields + num_players * 2,), dtype=np.int32)
        self.num_players = num_players
        self.batch_size = batch_size
        self.players = None  # np.zeros((num_players, 2))
        self.moves = np.zeros((5, 4), dtype=np.int32)
        self.moves[:, 1:3] = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)

    @staticmethod
    def from_str(data: str):
        num_players = sum(x.isdigit() for x in data)
        lines = data.split('\n')

        result = Pacman((len(lines), len(lines[0])), num_players, batch_size=1)
        for i, line in enumerate(lines):
            for j, c in enumerate(line):
                if c == '#':
                    result.grid[0, i, j, 1] = 1
                elif c == 's':
                    result.grid[0, i, j, 2] = 1
                elif c == 'S':
                    result.grid[0, i, j, 3] = 1
                elif c.isdigit():
                    result.grid[0, i, j, result.fields + int(c) - 1] = 1
                else:  # empty
                    result.grid[0, i, j, 0] = 1
        return result.copy_board(), result.size, result.num_players

    def copy_board(self):
        return np.copy(self.grid[:, :, :, :self.fields + self.num_players])

    def players_layer_shape(self):
        return self.size + (self.num_players,)

    def reset(self, data):
        """
        `data` must have (batch_size, height, width, 4) shape
        """
        self.grid[:, :, :, :self.fields + self.num_players] = data
        positions = np.where(self.grid[:, :, :, self.fields: self.fields + self.num_players] == 1.)
        # self.players holds [num_board, x, y, player]
        self.players = np.array(positions, dtype=np.int32).T  # CHECK: [:, 0] should be sorted, [:, 3] should be ranges
        return self.step(np.zeros((self.batch_size, self.num_players), dtype=np.int32))[0]

    def _not_blocked(self, positions):
        not_blocked = np.all(positions[:, 1:3] >= 0, axis=1)
        not_blocked &= np.all(positions[:, 1:3] < self.size, axis=1)
        pos_ins = positions[not_blocked]
        pos_ins[:, -1] = 1  # because 1 layer means walls
        not_blocked[not_blocked] = self.grid[list(pos_ins.T)] < 0.5
        return not_blocked

    def step(self, actions):
        new_pos = self.players + self.moves[actions.reshape(-1)]
        not_blocked = self._not_blocked(new_pos)

        reward = np.zeros((self.batch_size, self.num_players))
        if np.any(not_blocked):
            # remove players from grid
            positions = list(np.transpose(self.players + np.array([0, 0, 0, self.fields])))
            self.grid[positions] = 0

            # update players positions
            self.players[not_blocked] = new_pos[not_blocked]

            # add players to grid
            positions = list(np.transpose(self.players + np.array([0, 0, 0, self.fields])))
            self.grid[positions] = 1

            # calculate rewards
            small, large = np.copy(self.players), np.copy(self.players)
            small[:, -1] = 2  # because 2 layer means small-pellets
            small = list(small.T)  # get indices
            large[:, -1] = 3  # because 3 layer means small-pellets
            large = list(large.T)
            reward = 0.5 * self.grid[small] + self.grid[large]

            # split rewards between players
            uni, idx = np.unique(self.players[:, :3], return_inverse=True, axis=0)
            freq = np.bincount(idx)[idx]
            reward /= freq.astype(np.float32)

            # remove pellets from grid
            self.grid[small] = 0
            self.grid[large] = 0

        # generate new state for each player
        one_hot = np.eye(self.fields + 2 * self.num_players, dtype=np.float32)
        return tuple((self.grid + one_hot[self.grid_depth + player]).astype(np.float32)
                     for player in range(self.num_players)), tuple(reward.reshape(actions.shape).T)

    @property
    def actions(self):
        return 5  # [noop, up, down, left, right]

    def __repr__(self):
        """
        Shows only first game level.
        """

        def _field_repr(x, y):
            if self.grid[0, x, y, 1] == 1:  # wall
                return '#'
            if self.grid[0, x, y, 2] == 1:  # small reward
                return '•'
            if self.grid[0, x, y, 3] == 1:  # large reward
                return '♦'
            for p in range(self.num_players):
                if self.grid[0, x, y, self.fields + p] == 1:
                    return '{:1d}'.format(p)
            return ' '

        board = [[_field_repr(x, y)
                  for y in range(self.size[1])]
                 for x in range(self.size[0])]
        return '+' + '-' * self.size[0] + '+\n' + \
               '\n'.join('|' + ''.join(row) + '|' for row in board) + \
               '\n+' + '-' * self.size[0] + '+'

    @staticmethod
    def action_name(a):
        return ['noop', 'up', 'down', 'left', 'right'][a]
