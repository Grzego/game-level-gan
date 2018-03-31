import torch
from torch.autograd import Variable

from utils import cudify


class PytorchWrapper(object):
    """
    This wrapper is for correctly permute dimensions to PyTorch format.
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.env, attr)

    def __repr__(self):
        return self.env.__repr__()

    def _wrap_state(self, state):
        return Variable(cudify(torch.from_numpy(state))).permute(0, 3, 1, 2)

    def reset(self, base_board):
        if not isinstance(base_board, Variable):
            raise RuntimeError('base_board is not pytorch Variable!')
        return [self._wrap_state(state) for state in self.env.reset(base_board.cpu().data.numpy())]

    def step(self, actions):
        new_states, rewards = self.env.step(actions)
        return [self._wrap_state(state) for state in new_states], rewards
