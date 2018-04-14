import math
import torch

from utils.pytorch_utils import cudify
from .environment import MultiEnvironment


class RaceCar(object):
    def __init__(self, max_speed, max_angle):
        self.max_speed = max_speed  # speed in km/h
        self.max_angle = max_angle  # angle in degrees


class Race(MultiEnvironment):
    def __init__(self, timeout, cars: [RaceCar], framerate=1. / 30.):
        # 1unit = 10m
        # TODO: try continuous actions
        self.timeout = timeout
        self.framerate = framerate
        self.num_players = len(cars)
        self.cars = cars
        self.positions = None
        self.directions = None
        self.speeds = None
        self.bounds = None
        self.reward_bound = None

    def players_layer_shape(self):
        pass

    def reset(self, tracks):
        """
        tracks - [num_boards, segments, (arc, width)]

        where:
            arc is in range (-1, 1) meaning next segment angle from (-90deg, 90deg)
            width is in range (0, 1) meaning width from (0.5, 1.)
        """

        # add sentinels to tracks (0 in front and 0 in back of track for additional segment)
        num_boards = tracks.size(0)
        tracks = torch.cat((cudify(torch.zeros(num_boards, 2, 2)), tracks, cudify(torch.zeros(num_boards, 1, 2))),
                           dim=1)

        arcsum = 0.5 * math.pi * torch.cumsum(tracks[:, :, :1], dim=1)  # cumsum over angles and conversion to radians
        segment_vecs = torch.cat((torch.cos(arcsum), torch.sin(arcsum)), dim=2)

        signs = torch.sign(tracks[:, 1:, :1])
        signs[signs == 0.] = 1.

        right_vecs = -segment_vecs[:, :-1, :] + segment_vecs[:, 1:, :]
        right_vecs = right_vecs * signs / torch.sqrt(torch.sum(torch.pow(right_vecs, 2.), dim=-1, keepdim=True))
        right_vecs *= 0.5 + 0.5 * tracks[:, 1:, 1:]
        left_vecs = -right_vecs

        segments = torch.cumsum(segment_vecs[:, 1:, :], dim=1)
        right_vecs = segments + right_vecs
        left_vecs = segments + left_vecs

        right_bounds = torch.cat((right_vecs[:, :-1, :], right_vecs[:, 1:, :]), dim=-1)
        left_bounds = torch.cat((left_vecs[:, :-1, :], left_vecs[:, 1:, :]), dim=-1)
        start_bounds = torch.cat((left_vecs[:, :1, :], right_vecs[:, :1, :]), dim=-1)
        self.reward_bound = torch.cat((left_vecs[:, -1:, :], right_vecs[:, -1:, :]), dim=-1)
        self.bounds = torch.cat((right_bounds, left_bounds, start_bounds), dim=1)
        # TODO: players stating positions

    def step(self, actions):
        # TODO: 0) check if players is alive
        # TODO: 1) choose move vector (thats car and action dependent)
        # TODO: 2) update car position
        # TODO: 3) check for collisions
        # TODO: 4) check for reward
        # TODO: 5) return states (state is just distances in few directions)
        pass

    @property
    def actions(self):
        return 7  # [noop, forward, break, left, right, forward-left, forward-right]

    @staticmethod
    def action_name(a):
        return ['noop', 'forward', 'break', 'left', 'right', 'forward-left', 'forward-right'][a]
