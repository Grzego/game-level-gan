import torch
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')


def change_device(dev):
    global device
    device = dev


def one_hot(a, dim=-1, num_classes=None):
    """
    Encodes a to one-hot representation.
    `dim` which dimension should be expanded
    `num_classes` how many classes there is (if None then max value from `a` will be used)
    """
    if dim == -1:
        dim = len(a.shape)
    if num_classes is None:
        num_classes = torch.max(a)
    zeros = a.new_zeros(*a.shape[:dim], num_classes, *a.shape[dim:])
    return zeros.scatter_(dim, a.unsqueeze(dim), 1)


def gumbel_noise_like(tensor):
    """
    Requires `a` to be values before applying softmax.
    """
    gumbel = torch.empty_like(tensor)
    gumbel.uniform_()
    gumbel.add_(1e-8).log_().neg_()
    gumbel.add_(1e-8).log_().neg_()
    return gumbel


class Bipolar(nn.Module):
    def __init__(self, fn, dim=1):
        super().__init__()
        self.fn = fn
        self.dim = dim

    def forward(self, x):
        x0, x1 = torch.chunk(x, chunks=2, dim=self.dim)
        y0 = self.fn(x0)
        y1 = -self.fn(-x1)
        return torch.cat((y0, y1), dim=self.dim)
