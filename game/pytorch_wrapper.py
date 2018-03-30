import torch
from torch.autograd import Variable

from utils import cudify


class PytorchWrapper(object):
    """
    This wrapper is for extracting gradients w.r.t. base board
    so that generator can use those.
    """

    def __init__(self, env):
        self.env = env
        self.states = None
        self.base_board = None
        self.board = None

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.env, attr)

    def __repr__(self):
        return self.env.__repr__()

    def _wrap_state(self, state):
        diff = Variable(cudify(state - self.board.cpu().data))
        # permute dimensions and add batch dim
        state = self.board + diff
        return state.permute(0, 3, 1, 2)

    def reset(self, base_board):
        if not isinstance(base_board, Variable):
            raise RuntimeError('base_board is not pytorch Variable!')
        self.base_board = base_board
        player_layer = Variable(cudify(torch.zeros((base_board.size(0),) + self.env.players_layer_shape())))
        self.board = torch.cat((self.base_board, player_layer), dim=-1)
        return [self._wrap_state(state) for state in self.env.reset(base_board.cpu().data.numpy())]

    def step(self, actions):
        new_states, rewards = self.env.step(actions)
        return [self._wrap_state(state) for state in new_states], rewards

    @property
    def actions(self):
        return self.env.actions
