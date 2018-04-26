import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


def gumbel_noise(shape):
    """
    Requires `a` to be values before applying softmax.
    """
    gumbel = torch.rand(*shape, device=device)
    gumbel.add_(1e-8).log_().neg_()
    gumbel.add_(1e-8).log_().neg_()
    return gumbel
