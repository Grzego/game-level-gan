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
    if dim == -1:
        dim = len(a.shape)
    if num_classes is None:
        num_classes = torch.max(a)
    zeros = cudify(torch.zeros(*a.shape[:dim], num_classes, *a.shape[dim:]), use_cuda=a.is_cuda)
    is_var = isinstance(a, Variable)
    if is_var:
        a = a.data
    encoded = zeros.scatter_(dim, a.unsqueeze(dim), 1)
    return Variable(encoded) if is_var else encoded


def gumbel_noise(shape):
    """
    Requires `a` to be values before applying softmax.
    """
    gumbel = cudify(torch.rand(*shape))
    gumbel.add_(1e-8).log_().neg_()
    gumbel.add_(1e-8).log_().neg_()
    return gumbel
