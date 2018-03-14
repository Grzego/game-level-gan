import torch
from torch.autograd import Variable


def cudify(a, use_cuda=False):
    """
    Moves tensor/Variable/Module to cuda if available.
    """
    if torch.cuda.is_available() or use_cuda:
        return a.cuda()
    return a


def one_hot(a, dim=-1, num_classes=None):
    """
    Encodes a to one-hot representation.
    `dim` which dimension should be expanded
    `num_classes` how many classes there is (if None then max value from `a` will be used)
    """
    if num_classes is None:
        num_classes = torch.max(a)
    zeros = cudify(torch.zeros(*a.shape[:dim], num_classes, *a.shape[dim:]), use_cuda=a.is_cuda)
    if isinstance(a, Variable):
        a = a.data
    return zeros.scatter_(dim, a.unsqueeze(dim), 1)
