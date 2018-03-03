import numpy as np
import torch
from torch.autograd import Variable


class PytorchWrapper(object):
    """
    This wrapper is for extracting gradients w.r.t. base board
    so that generator can use those.
    """

    def __init__(self, env, use_cuda=True):
        self.env = env
        self.use_cuda = use_cuda
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
        diff = Variable(state - self.board.cpu().data)
        diff = diff.cuda() if self.use_cuda else diff
        # permute dimensions and add batch dim
        state = self.board + diff
        return state.permute(2, 0, 1)[None, ...]

    def reset(self, base_board):
        self.base_board = Variable(torch.from_numpy(base_board.astype(np.float32)), requires_grad=True)
        base_board = self.base_board.cuda() if self.use_cuda else self.base_board
        player_layer = Variable(torch.zeros(self.env.players_layer_shape()))
        player_layer = player_layer.cuda() if self.use_cuda else player_layer
        self.board = torch.cat((base_board, player_layer), dim=-1)
        return [self._wrap_state(state) for state in self.env.reset(base_board)]

    def step(self, actions):
        new_states, rewards = self.env.step(actions)
        return [self._wrap_state(state) for state in new_states], rewards

    def actions(self):
        return self.env.actions()
