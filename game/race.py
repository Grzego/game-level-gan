import math
import torch
import numpy as np

from utils.pytorch_utils import cudify, tensor_from_list
from .environment import MultiEnvironment


class RaceCar(object):
    def __init__(self, max_speed, acceleration, angle):
        # max_speed in km/h
        # max_acceleration in m/s^2
        # max_angle in degrees/s

        # converting to units/s, units/s^2 and rads/s
        self.max_speed = max_speed * 100. / 3600.
        self.acceleration = acceleration * 0.1
        self.angle = angle * math.pi / 180.


class Race(MultiEnvironment):
    def __init__(self, timeout, cars: [RaceCar], observation_size=9, max_distance=2., framerate=1. / 30.):
        # 1unit = 10m
        # TODO: try continuous actions
        self.timeout = timeout
        self.framerate = framerate
        self.num_players = len(cars)
        self.cars_max_speed = tensor_from_list([car.max_speed for car in cars], dtype=np.float32)
        self.cars_acceleration = tensor_from_list([car.acceleration for car in cars], dtype=np.float32)
        self.cars_angle = tensor_from_list([car.angle for car in cars], dtype=np.float32)
        self.positions = None
        self.directions = None
        self.speeds = None
        self.alive = None
        self.steps = 0
        self.steps_limit = int(timeout // framerate)
        self.bounds = None
        self.reward_bound = None
        self.action_speed = tensor_from_list([0.,   # noop
                                              1.,   # forward
                                              -1.,  # backward
                                              0.,   # left
                                              0.,   # right
                                              1.,   # forward-left
                                              1.,   # forward-right
                                              -1.,  # backward-left
                                              -1.,  # backward-right
                                              ], dtype=np.float32)
        self.action_dirs = tensor_from_list([0.,   # noop
                                             0.,   # forward
                                             0.,   # backward
                                             1.,   # left
                                             -1.,  # right
                                             1.,   # forward-left
                                             -1.,  # forward-right
                                             1.,   # backward-left
                                             -1.,  # backward-right
                                             ], dtype=np.float32)
        self.observation_size = observation_size
        self.max_distance = max_distance

    def state_shape(self):
        return self.observation_size,

    def players_layer_shape(self):
        pass

    def reset(self, tracks):
        """
        tracks - [num_boards, segments, (arc, width)]

        where:
            arc is in range (-1, 1) meaning next segment angle from (-90deg, 90deg)
            width is in range (0, 1) meaning width from (0.5, 1.)
        """
        self.steps = 0

        # add sentinels to tracks (0 in front and 0 in back of track for additional segment)
        num_boards = tracks.size(0)
        tracks = torch.cat((cudify(torch.zeros(num_boards, 1, 2)), tracks, cudify(torch.zeros(num_boards, 1, 2))),
                           dim=1)

        arcsum = 0.5 * math.pi * torch.cumsum(tracks[:, :, :1], dim=1)  # cumsum over angles and conversion to radians
        segment_vecs = torch.cat((torch.sin(arcsum), torch.cos(arcsum)), dim=2)

        right_vecs = segment_vecs.clone()
        right_vecs[:, :, 0], right_vecs[:, :, 1] = segment_vecs[:, :, 1], -segment_vecs[:, :, 0]
        right_vecs *= 0.5 + 0.5 * tracks[:, :, 1:]
        left_vecs = -right_vecs

        segments = torch.cumsum(segment_vecs, dim=1)
        right_vecs = segments + right_vecs
        left_vecs = segments + left_vecs

        right_bounds = torch.cat((right_vecs[:, :-1, :], right_vecs[:, 1:, :]), dim=-1)
        left_bounds = torch.cat((left_vecs[:, :-1, :], left_vecs[:, 1:, :]), dim=-1)
        start_bounds = torch.cat((left_vecs[:, :1, :], right_vecs[:, :1, :]), dim=-1)
        reward_bound = torch.cat((left_vecs[:, -1:, :], right_vecs[:, -1:, :]), dim=-1)
        reward_bound = reward_bound.repeat(1, self.num_players, 1, 1)
        self.reward_bound = reward_bound.view(-1, *reward_bound.shape[-2:])
        bounds = torch.cat((right_bounds, left_bounds, start_bounds), dim=1).unsqueeze(1)
        bounds = bounds.repeat(1, self.num_players, 1, 1)
        self.bounds = bounds.view(-1, *bounds.shape[-2:])  # [num_boards * num_players, num_segments, 4]

        self.positions = cudify(torch.zeros(num_boards, self.num_players, 2))
        self.positions[:, :, 1] = 1.1  # 1m after a start
        self.directions = cudify(torch.zeros(num_boards, self.num_players, 2))
        self.directions[:, :, 1] = 1.  # direction vectors should have length = 1.
        self.speeds = cudify(torch.zeros(num_boards, self.num_players))
        self.alive = cudify(torch.ByteTensor(num_boards, self.num_players))
        self.alive.fill_(True)

        return self.step(cudify(torch.zeros(num_boards, self.num_players).long()))[0]

    @staticmethod
    def _segment_collisions(segments, tests, special=False):
        """
        Tests collision between every S and P for every N.
        Returns True/False wherther collision occured (shape [N, S, P])

        segments = [N, S, 4]
        tests = [N, P, 4]

        N - usually batch size or number of boards
        S - number of segments
        P - number of players
        """
        n, s, p = segments.size(0), segments.size(1), tests.size(1)

        def _orientation(p, q, r):
            """
            Returns orientation of r point with respect to segment p -> q
            (-1, 0, 1) - (widdershins - counterclockwise, colinear, clockwise)
            """
            p, q, r = p.unsqueeze(2), q.unsqueeze(2), r.unsqueeze(1)
            qp = q - p
            rq = r - q
            return torch.sign(qp[:, :, :, 1] * rq[:, :, :, 0] - qp[:, :, :, 0] * rq[:, :, :, 1])

        def _on_segment(p, q, r):
            """
            Checks whether point r lies on p -> q segment
            """
            p, q, r = p.unsqueeze(2), q.unsqueeze(2), r.unsqueeze(1)
            return torch.prod((r <= torch.max(p, q)) & (r >= torch.min(p, q)), dim=-1)

        p1, q1 = segments[:, :, :2], segments[:, :, 2:]
        p2, q2 = tests[:, :, :2], tests[:, :, 2:]

        o1 = _orientation(p1, q1, p2).view(n, -1)  # [N, S]
        o2 = _orientation(p1, q1, q2).view(n, -1)  # [N, S]
        o3 = _orientation(p2, q2, p1).permute(0, 2, 1).contiguous().view(n, -1)  # [N, S] - permute to match cases
        o4 = _orientation(p2, q2, q1).permute(0, 2, 1).contiguous().view(n, -1)  # [N, S]

        # general case
        res = (o1 != o2) & (o3 != o4)

        # special cases
        spec = (o1 == 0.) & _on_segment(p1, q1, p2).view(n, -1)
        res |= (o2 == 0.) & _on_segment(p1, q1, q2).view(n, -1)
        res |= (o3 == 0.) & _on_segment(p2, q2, p1).permute(0, 2, 1).contiguous().view(n, -1)
        res |= (o4 == 0.) & _on_segment(p2, q2, q1).permute(0, 2, 1).contiguous().view(n, -1)

        return (res | spec).view(n, s, p) if not special else (res.view(n, s, p), spec.view(n, s, p))

    @staticmethod
    def _smallest_distance(segments, directions):
        """
        Returns smallest distances to any segments in given `directions`.

        segments = [N, S, 4]
        directions = [N, D, 4]

        N - usually batch size or number of boards
        S - number of segments
        D - number of distinct directions

        Returns tensor with shape [N, D]
        """
        n, d, s = directions.size(0), directions.size(1), segments.size(1)

        # get collisions mask
        far_dirs = directions.clone()
        far_dirs[:, :, 2:] = directions[:, :, :2] + 1000. * directions[:, :, 2:]
        collisions, start_on_segment = Race._segment_collisions(segments, far_dirs, special=True)
        collisions &= ~start_on_segment

        p, q = segments[:, :, None, :2], segments[:, :, None, 2:]
        s, d = directions[:, None, :, :2], directions[:, None, :, 2:]

        qpc = q - p  # [N, S, 1, 2]
        spc = p - s  # [N, S, D, 2]

        dists = spc[:, :, :, 1] * qpc[:, :, :, 0] - spc[:, :, :, 0] * qpc[:, :, :, 1]  # [N, S, D]
        denom = d[:, :, :, 1] * qpc[:, :, :, 0] - d[:, :, :, 0] * qpc[:, :, :, 1]  # [N, S, D]

        dists[start_on_segment] = 0.
        dists[collisions] /= denom[collisions]
        dists[~(collisions | start_on_segment)] = float('inf')
        dists[dists < 0.] = float('inf')

        return torch.min(dists, dim=1)[0]

    @staticmethod
    def _rotate_vecs(vectors, angles):
        """
        Rotates vectors to the left by given angle.

        vectors = [N, 2]
        angles = [N]
        """
        angles = angles.view(-1, 1).repeat(1, 4).view(-1, 2, 2)
        angles[:, 0, 0].cos_()
        angles[:, 0, 1].sin_().neg_()
        angles[:, 1, 0].sin_()
        angles[:, 1, 1].cos_()

        return torch.matmul(vectors.unsqueeze(1), angles).view(-1, 2)

    def step(self, actions):
        """
        actions - torch.Tensor with size [num_boards, num_players]
        """
        self.steps += 1
        num_boards, num_seg = actions.size(0), self.bounds.size(2)

        #  choose move vector (thats car and action dependent)
        dir_flag = self.action_dirs[actions.view(-1)]  # [num_boards * num_players]
        angles = self.framerate * dir_flag * self.cars_angle.repeat(num_boards).view(-1)  # [num_boards * num_players]
        new_dirs = self._rotate_vecs(self.directions.view(-1, 2), angles).view(-1, self.num_players, 2)

        speed_flag = self.action_speed[actions.view(-1)].view(num_boards, -1)  # [num_boards, num_players]
        speed = self.speeds + self.framerate * speed_flag * self.cars_acceleration[None, :]
        new_speed = torch.min(self.cars_max_speed, torch.max(self.cars_max_speed.neg(), speed))
        is_moving = torch.abs(new_speed.view(-1)) > 1e-7  # [num_boards * num_players]

        new_pos = self.positions + new_dirs * new_speed[:, :, None]

        # pick boards with alive players
        update_mask = (self.alive.view(-1) & is_moving).nonzero().squeeze()

        rewards = cudify(torch.FloatTensor(num_boards * self.num_players))
        rewards.fill_(-0.01)  # small negative reward over time

        if torch.sum(is_moving) > 0:  # anyone moved
            # check collisions
            paths = torch.cat((self.positions, new_pos), dim=-1).view(-1, 1, 4)  # [num_boards * num_players, 1, 4]

            seg_col = self._segment_collisions(self.bounds[update_mask], paths[update_mask])
            is_dead = seg_col.squeeze().max(dim=1)[0]
            self.alive.view(-1)[update_mask] = ~is_dead

            #  check for reward
            # +1 if finished
            # -1 if died
            # -0.01 otherwise to encourage finishing race faster
            finish = self._segment_collisions(self.reward_bound[update_mask], paths[update_mask])
            is_done = torch.max(finish.view(update_mask.size(0), -1), dim=-1)[0]
            rewards[update_mask] += is_done.float() - is_dead.float()

        # update all player variables
        self.directions = new_dirs
        self.speeds = new_speed
        self.positions = new_pos

        # return states (state is just distances in few directions)
        obs_angles = torch.linspace(-math.pi / 2., math.pi / 2., self.observation_size).view(1, -1)
        obs_angles = obs_angles.repeat(num_boards * self.num_players, 1).view(-1)
        rot_dirs = new_dirs.view(-1, 1, 2).repeat(1, self.observation_size, 1).view(-1, 2)
        obs_dirs = self._rotate_vecs(rot_dirs, obs_angles)
        obs_dirs = obs_dirs.view(-1, self.observation_size, 2)  # [num_boards * num_players, observation_size, 2]
        obs_segm = new_pos.view(-1, 1, 2).repeat(1, self.observation_size, 1)
        obs_segm = torch.cat((obs_segm, obs_dirs), dim=-1)

        states = cudify(torch.zeros(num_boards * self.num_players, self.observation_size))
        alive_mask = self.alive.view(-1).nonzero().squeeze()
        alive_states = self._smallest_distance(self.bounds[alive_mask], obs_segm[alive_mask])
        states[alive_mask] = alive_states.clamp(max=self.max_distance)

        return states.view(num_boards, self.num_players, -1).permute(1, 0, 2), rewards.view(num_boards, -1).t()

    def finished(self):
        #      all players are crushed     or timeout was reached
        return torch.sum(self.alive) < 0.1 or self.steps > self.steps_limit

    @property
    def actions(self):
        return 9

    @staticmethod
    def action_name(a):
        return ('noop', 'forward', 'backward', 'left', 'right',
                'forward-left', 'forward-right', 'backward-left', 'backward-right')[a]
