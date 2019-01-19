import torch
from torch.utils import cpp_extension as ext


def main():
    game = ext.load('game', sources=['game_helpers.cpp'], extra_cflags=['-O3', '-DNDEBUG', '-fopenmp'],
                    extra_ldflags=['-lpthread'])

    valid_left = torch.zeros(2, 4, 2)
    valid_left[:, :, 0] = -1.
    valid_left[:, :, 1] = torch.linspace(0., 3., 4)[None, :].repeat(2, 1)
    valid_right = valid_left.clone()
    valid_right[:, :, 0] = 1.

    gg = game.Game(valid_left, valid_right, 2)
    print(gg.validate_tracks())

    idx = torch.tensor([0])
    dirs = torch.tensor([[[0., 1., -1., 0.],
                          [0., 1., 0., 1.],
                          [0., 1., 1., 0],
                          [0., 1., 0, -1]]])
    print(gg.smallest_distance(idx, dirs))

    idx = torch.tensor([0])
    new_pos = torch.tensor([[0., 10.]])
    print(gg.update_players(idx, new_pos))


if __name__ == '__main__':
    main()
